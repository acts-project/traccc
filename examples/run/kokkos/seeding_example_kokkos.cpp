/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/efficiency/seeding_performance_writer.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_spacepoints.hpp"
#include "traccc/kokkos/seeding/details/spacepoint_binning.hpp"
#include "traccc/options/accelerator.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_fitting.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_seeding.hpp"
#include "traccc/performance/collection_comparator.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>

// Kokkos include(s).
#include <Kokkos_Core.hpp>

int seq_run(const traccc::opts::track_seeding& seeding_opts,
            const traccc::opts::track_finding& /*finding_opts*/,
            const traccc::opts::track_propagation& /*propagation_opts*/,
            const traccc::opts::track_fitting& /*fitting_opts*/,
            const traccc::opts::input_data& input_opts,
            const traccc::opts::detector& detector_opts,
            const traccc::opts::performance& performance_opts,
            const traccc::opts::accelerator& accelerator_opts,
            std::unique_ptr<const traccc::Logger> ilogger) {
    TRACCC_LOCAL_LOGGER(std::move(ilogger));

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;
    traccc::memory_resource mr{host_mr, &host_mr};

    // Construct a Detray detector object, if supported by the configuration.
    traccc::default_detector::host host_det{host_mr};
    assert(detector_opts.use_detray_detector == true);
    traccc::io::read_detector(host_det, host_mr, detector_opts.detector_file,
                              detector_opts.material_file,
                              detector_opts.grid_file);

    traccc::host::seeding_algorithm sa(
        seeding_opts.seedfinder, {seeding_opts.seedfinder},
        seeding_opts.seedfilter, host_mr, logger().clone("HostSeedingAlg"));
    traccc::host::track_params_estimation tp(
        host_mr, logger().clone("HostTrackParEstAlg"));

    // Output stats
    uint64_t n_spacepoints = 0;
    uint64_t n_seeds = 0;
    uint64_t n_seeds_kokkos = 0;

    // KOKKOS Spacepoint Binning
    traccc::kokkos::details::spacepoint_binning m_spacepoint_binning(
        seeding_opts.seedfinder, {seeding_opts.seedfinder}, mr,
        logger().clone("KokkosBinningAlg"));

    // performance writer
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{});

    traccc::performance::timing_info elapsedTimes;

    // Loop over events
    for (std::size_t event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        traccc::measurement_collection_types::host measurements_per_event{
            &host_mr};
        traccc::edm::spacepoint_collection::host spacepoints_per_event{host_mr};
        traccc::host::seeding_algorithm::output_type seeds{host_mr};
        traccc::host::track_params_estimation::output_type params{&host_mr};

        {  // Start measuring wall time
            traccc::performance::timer wall_t("Wall time", elapsedTimes);

            /*-----------------
            hit file reading
            -----------------*/
            {
                traccc::performance::timer t("Hit reading  (cpu)",
                                             elapsedTimes);
                // Read the hits from the relevant event file
                traccc::io::read_spacepoints(
                    spacepoints_per_event, measurements_per_event, event,
                    input_opts.directory,
                    (input_opts.use_acts_geom_source ? &host_det : nullptr),
                    input_opts.format);
            }  // stop measuring hit reading timer

            {  // Spacepoin binning for kokkos
                traccc::performance::timer t("Spacepoint binning (kokkos)",
                                             elapsedTimes);
                m_spacepoint_binning(vecmem::get_data(spacepoints_per_event));
            }

            /*----------------------------
                Seeding algorithm
            ----------------------------*/

            // CPU

            if (accelerator_opts.compare_with_cpu) {
                traccc::performance::timer t("Seeding  (cpu)", elapsedTimes);
                seeds = sa(vecmem::get_data(spacepoints_per_event));
            }  // stop measuring seeding cpu timer

            /*----------------------------
            Track params estimation
            ----------------------------*/

            // CPU

            if (accelerator_opts.compare_with_cpu) {
                traccc::performance::timer t("Track params  (cpu)",
                                             elapsedTimes);
                params = tp(vecmem::get_data(measurements_per_event),
                            vecmem::get_data(spacepoints_per_event),
                            vecmem::get_data(seeds),
                            {0.f, 0.f, seeding_opts.seedfinder.bFieldInZ});
            }  // stop measuring track params cpu timer

        }  // Stop measuring wall time

        /*----------------
             Statistics
          ---------------*/

        n_spacepoints += spacepoints_per_event.size();
        n_seeds += seeds.size();
    }

    if (performance_opts.run) {
        sd_performance_writer.finalize();
    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_spacepoints << " spacepoints" << std::endl;
    std::cout << "- created (cpu)  " << n_seeds << " seeds" << std::endl;
    std::cout << "- created (kokkos) " << n_seeds_kokkos << " seeds"
              << std::endl;
    std::cout << "==>Elapsed times...\n" << elapsedTimes << std::endl;

    return 0;
}

// The main routine
//
int main(int argc, char* argv[]) {
    std::unique_ptr<const traccc::Logger> logger = traccc::getDefaultLogger(
        "TracccExampleSeqKokkos", traccc::Logging::Level::INFO);

    // Initialise both Kokkos.
    Kokkos::initialize(argc, argv);

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::track_seeding seeding_opts;
    traccc::opts::track_finding finding_opts;
    traccc::opts::track_propagation propagation_opts;
    traccc::opts::track_fitting fitting_opts;
    traccc::opts::performance performance_opts;
    traccc::opts::accelerator accelerator_opts;
    traccc::opts::program_options program_opts{
        "Full Tracking Chain Using Kokkos (without clusterization)",
        {detector_opts, input_opts, seeding_opts, finding_opts,
         propagation_opts, fitting_opts, performance_opts, accelerator_opts},
        argc,
        argv,
        logger->cloneWithSuffix("Options")};

    // Run the application.
    const int ret = seq_run(
        seeding_opts, finding_opts, propagation_opts, fitting_opts, input_opts,
        detector_opts, performance_opts, accelerator_opts, logger->clone());

    // Finalise Kokkos.
    Kokkos::finalize();

    return ret;
}
