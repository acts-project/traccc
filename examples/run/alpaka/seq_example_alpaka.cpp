/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/alpaka/clusterization/clusterization_algorithm.hpp"
#include "traccc/alpaka/clusterization/measurement_sorting_algorithm.hpp"
#include "traccc/alpaka/seeding/seeding_algorithm.hpp"
#include "traccc/alpaka/seeding/spacepoint_formation_algorithm.hpp"
#include "traccc/alpaka/seeding/track_params_estimation.hpp"
#include "traccc/alpaka/utils/vecmem_types.hpp"
#ifdef ALPAKA_ACC_SYCL_ENABLED
#include <sycl/sycl.hpp>
#include <vecmem/utils/sycl/queue_wrapper.hpp>
#endif

#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/efficiency/seeding_performance_writer.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_detector_description.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/options/accelerator.hpp"
#include "traccc/options/clusterization.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_seeding.hpp"
#include "traccc/performance/collection_comparator.hpp"
#include "traccc/performance/soa_comparator.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/silicon_pixel_spacepoint_formation_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// System include(s).
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>

namespace po = boost::program_options;

int seq_run(const traccc::opts::detector& detector_opts,
            const traccc::opts::input_data& input_opts,
            const traccc::opts::clusterization& clusterization_opts,
            const traccc::opts::track_seeding& seeding_opts,
            const traccc::opts::performance& performance_opts,
            const traccc::opts::accelerator& accelerator_opts,
            std::unique_ptr<const traccc::Logger> ilogger) {
    TRACCC_LOCAL_LOGGER(std::move(ilogger));

    // Output stats
    uint64_t n_cells = 0;
    uint64_t n_measurements = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_spacepoints_alpaka = 0;
    uint64_t n_seeds = 0;
    uint64_t n_seeds_alpaka = 0;

    // Constant B field for the track finding and fitting
    const traccc::vector3 field_vec = {0.f, 0.f,
                                       seeding_opts.seedfinder.bFieldInZ};

    // Memory resources used by the application.
#ifdef ALPAKA_ACC_SYCL_ENABLED
    ::sycl::queue q;
    vecmem::sycl::queue_wrapper qw{&q};
    traccc::alpaka::vecmem::device_copy copy(qw);
    traccc::alpaka::vecmem::host_memory_resource host_mr(qw);
    traccc::alpaka::vecmem::device_memory_resource device_mr(qw);
#else
    traccc::alpaka::vecmem::device_copy copy;
    traccc::alpaka::vecmem::host_memory_resource host_mr;
    traccc::alpaka::vecmem::device_memory_resource device_mr;
#endif
    traccc::memory_resource mr{device_mr, &host_mr};

    // Construct the detector description object.
    traccc::silicon_detector_description::host host_det_descr{host_mr};
    traccc::io::read_detector_description(
        host_det_descr, detector_opts.detector_file,
        detector_opts.digitization_file,
        (detector_opts.use_detray_detector ? traccc::data_format::json
                                           : traccc::data_format::csv));
    traccc::silicon_detector_description::data host_det_descr_data{
        vecmem::get_data(host_det_descr)};
    traccc::silicon_detector_description::buffer device_det_descr{
        static_cast<traccc::silicon_detector_description::buffer::size_type>(
            host_det_descr.size()),
        mr.main};
    copy(host_det_descr_data, device_det_descr)->wait();

    // Construct a Detray detector object, if supported by the configuration.
    traccc::default_detector::host host_detector{host_mr};
    traccc::default_detector::buffer device_detector;
    traccc::default_detector::view device_detector_view;
    if (detector_opts.use_detray_detector) {
        traccc::io::read_detector(
            host_detector, host_mr, detector_opts.detector_file,
            detector_opts.material_file, detector_opts.grid_file);
        device_detector = detray::get_buffer(host_detector, mr.main, copy);
        device_detector_view = detray::get_data(device_detector);
    }

    // Type definitions
    using host_spacepoint_formation_algorithm =
        traccc::host::silicon_pixel_spacepoint_formation_algorithm;
    using device_spacepoint_formation_algorithm =
        traccc::alpaka::spacepoint_formation_algorithm<
            traccc::default_detector::device>;

    traccc::host::clusterization_algorithm ca(
        host_mr, logger().clone("HostClusteringAlg"));
    host_spacepoint_formation_algorithm sf(
        host_mr, logger().clone("HostSpFormationAlg"));
    traccc::host::seeding_algorithm sa(
        seeding_opts.seedfinder, {seeding_opts.seedfinder},
        seeding_opts.seedfilter, host_mr, logger().clone("HostSeedingAlg"));
    traccc::host::track_params_estimation tp(
        host_mr, logger().clone("HostTrackParEstAlg"));

    traccc::alpaka::clusterization_algorithm ca_alpaka(
        mr, copy, clusterization_opts, logger().clone("AlpakaClusteringAlg"));
    traccc::alpaka::measurement_sorting_algorithm ms_alpaka(
        copy, logger().clone("AlpakaMeasSortingAlg"));
    device_spacepoint_formation_algorithm sf_alpaka(
        mr, copy, logger().clone("AlpakaSpFormationAlg"));
    traccc::alpaka::seeding_algorithm sa_alpaka(
        seeding_opts.seedfinder, {seeding_opts.seedfinder},
        seeding_opts.seedfilter, mr, copy, logger().clone("AlpakaSeedingAlg"));
    traccc::alpaka::track_params_estimation tp_alpaka(
        mr, copy, logger().clone("AlpakaTrackParEstAlg"));

    // performance writer
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{});

    traccc::performance::timing_info elapsedTimes;

    // Loop over events
    for (std::size_t event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        // Instantiate host containers/collections
        traccc::host::clusterization_algorithm::output_type
            measurements_per_event;
        host_spacepoint_formation_algorithm::output_type spacepoints_per_event{
            host_mr};
        traccc::host::seeding_algorithm::output_type seeds{host_mr};
        traccc::host::track_params_estimation::output_type params{&host_mr};

        // Instantiate alpaka containers/collections
        traccc::measurement_collection_types::buffer measurements_alpaka_buffer(
            0, *mr.host);
        traccc::edm::spacepoint_collection::buffer spacepoints_alpaka_buffer;
        traccc::edm::seed_collection::buffer seeds_alpaka_buffer;
        traccc::bound_track_parameters_collection_types::buffer
            params_alpaka_buffer(0, *mr.host);

        {
            traccc::performance::timer wall_t("Wall time", elapsedTimes);

            traccc::edm::silicon_cell_collection::host cells_per_event{host_mr};

            {
                traccc::performance::timer t("File reading  (cpu)",
                                             elapsedTimes);
                // Read the cells from the relevant event file into host memory.
                static constexpr bool DEDUPLICATE = true;
                traccc::io::read_cells(
                    cells_per_event, event, input_opts.directory,
                    logger().clone(), &host_det_descr, input_opts.format,
                    DEDUPLICATE, input_opts.use_acts_geom_source);
            }  // stop measuring file reading timer

            n_cells += cells_per_event.size();

            // Create device copy of input collections
            traccc::edm::silicon_cell_collection::buffer cells_buffer(
                static_cast<unsigned int>(cells_per_event.size()), mr.main);
            copy(vecmem::get_data(cells_per_event), cells_buffer)->wait();

            // Alpaka
            {
                traccc::performance::timer t("Clusterization (alpaka)",
                                             elapsedTimes);
                // Reconstruct it into spacepoints on the device.
                measurements_alpaka_buffer =
                    ca_alpaka(cells_buffer, device_det_descr);
                ms_alpaka(measurements_alpaka_buffer);
            }  // stop measuring clusterization alpaka timer

            // CPU
            if (accelerator_opts.compare_with_cpu) {
                traccc::performance::timer t("Clusterization  (cpu)",
                                             elapsedTimes);
                measurements_per_event =
                    ca(vecmem::get_data(cells_per_event), host_det_descr_data);
            }  // stop measuring clusterization cpu timer

            if (detector_opts.use_detray_detector) {

                // Alpaka
                {
                    traccc::performance::timer t(
                        "Spacepoint formation (alpaka)", elapsedTimes);
                    spacepoints_alpaka_buffer = sf_alpaka(
                        device_detector_view, measurements_alpaka_buffer);
                }  // stop measuring spacepoint formation cuda timer

                // CPU
                if (accelerator_opts.compare_with_cpu) {
                    traccc::performance::timer t("Spacepoint formation  (cpu)",
                                                 elapsedTimes);
                    spacepoints_per_event =
                        sf(host_detector,
                           vecmem::get_data(measurements_per_event));
                }  // stop measuring spacepoint formation cpu timer

                // Alpaka
                {
                    traccc::performance::timer t("Seeding (alpaka)",
                                                 elapsedTimes);
                    seeds_alpaka_buffer = sa_alpaka(spacepoints_alpaka_buffer);
                }  // stop measuring seeding alpaka timer

                // CPU
                if (accelerator_opts.compare_with_cpu) {
                    traccc::performance::timer t("Seeding  (cpu)",
                                                 elapsedTimes);
                    seeds = sa(vecmem::get_data(spacepoints_per_event));
                }  // stop measuring seeding cpu timer

                // Alpaka
                {
                    traccc::performance::timer t("Track params (alpaka)",
                                                 elapsedTimes);
                    params_alpaka_buffer = tp_alpaka(
                        measurements_alpaka_buffer, spacepoints_alpaka_buffer,
                        seeds_alpaka_buffer, field_vec);
                }  // stop measuring track params timer

                // CPU
                if (accelerator_opts.compare_with_cpu) {
                    traccc::performance::timer t("Track params  (cpu)",
                                                 elapsedTimes);
                    params = tp(vecmem::get_data(measurements_per_event),
                                vecmem::get_data(spacepoints_per_event),
                                vecmem::get_data(seeds), field_vec);
                }  // stop measuring track params cpu timer
            }
        }  // Stop measuring wall time

        /*----------------------------------
          compare cpu and alpaka result
          ----------------------------------*/

        traccc::edm::spacepoint_collection::host spacepoints_per_event_alpaka{
            host_mr};
        traccc::edm::seed_collection::host seeds_alpaka{host_mr};
        traccc::bound_track_parameters_collection_types::host params_alpaka{
            &host_mr};

        copy(spacepoints_alpaka_buffer, spacepoints_per_event_alpaka)->wait();
        copy(seeds_alpaka_buffer, seeds_alpaka)->wait();
        copy(params_alpaka_buffer, params_alpaka)->wait();

        if (accelerator_opts.compare_with_cpu) {

            // Show which event we are currently presenting the results for.
            std::cout << "===>>> Event " << event << " <<<===" << std::endl;

            // Compare the spacepoints made on the host and on the device.
            traccc::soa_comparator<traccc::edm::spacepoint_collection>
                compare_spacepoints{"spacepoints"};
            compare_spacepoints(vecmem::get_data(spacepoints_per_event),
                                vecmem::get_data(spacepoints_per_event_alpaka));

            // Compare the seeds made on the host and on the device
            traccc::soa_comparator<traccc::edm::seed_collection> compare_seeds{
                "seeds", traccc::details::comparator_factory<
                             traccc::edm::seed_collection::const_device::
                                 const_proxy_type>{
                             vecmem::get_data(spacepoints_per_event),
                             vecmem::get_data(spacepoints_per_event_alpaka)}};
            compare_seeds(vecmem::get_data(seeds),
                          vecmem::get_data(seeds_alpaka));

            // Compare the track parameters made on the host and on the device.
            traccc::collection_comparator<traccc::bound_track_parameters<>>
                compare_track_parameters{"track parameters"};
            compare_track_parameters(vecmem::get_data(params),
                                     vecmem::get_data(params_alpaka));
        }

        /// Statistics
        n_measurements += measurements_per_event.size();
        n_spacepoints += spacepoints_per_event.size();
        n_seeds += seeds.size();
        n_spacepoints_alpaka += spacepoints_per_event_alpaka.size();
        n_seeds_alpaka += seeds_alpaka.size();

        if (performance_opts.run) {

            traccc::event_data evt_data(input_opts.directory, event, host_mr,
                                        input_opts.use_acts_geom_source,
                                        &host_detector, input_opts.format,
                                        false);

            sd_performance_writer.write(
                vecmem::get_data(seeds),
                vecmem::get_data(spacepoints_per_event),
                vecmem::get_data(measurements_per_event), evt_data);
        }
    }

    if (performance_opts.run) {
        sd_performance_writer.finalize();
    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_cells << " cells" << std::endl;
    std::cout << "- created (cpu)  " << n_measurements << " measurements     "
              << std::endl;
    std::cout << "- created (cpu)  " << n_spacepoints << " spacepoints     "
              << std::endl;
    std::cout << "- created (alpaka) " << n_spacepoints_alpaka
              << " spacepoints     " << std::endl;

    std::cout << "- created  (cpu) " << n_seeds << " seeds" << std::endl;
    std::cout << "- created (alpaka) " << n_seeds_alpaka << " seeds"
              << std::endl;
    std::cout << "==>Elapsed times...\n" << elapsedTimes << std::endl;

    return 0;
}

// The main routine
//
int main(int argc, char* argv[]) {
    std::unique_ptr<const traccc::Logger> logger = traccc::getDefaultLogger(
        "TracccExampleSeqAlpaka", traccc::Logging::Level::INFO);

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::clusterization clusterization_opts;
    traccc::opts::track_seeding seeding_opts;
    traccc::opts::performance performance_opts;
    traccc::opts::accelerator accelerator_opts;
    traccc::opts::program_options program_opts{
        "Full Tracking Chain Using Alpaka",
        {detector_opts, input_opts, clusterization_opts, seeding_opts,
         performance_opts, accelerator_opts},
        argc,
        argv,
        logger->cloneWithSuffix("Options")};

    // Run the application.
    return seq_run(detector_opts, input_opts, clusterization_opts, seeding_opts,
                   performance_opts, accelerator_opts, logger->clone());
}
