/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/clusterization/measurement_sorting_algorithm.hpp"
#include "traccc/cuda/finding/finding_algorithm.hpp"
#include "traccc/cuda/fitting/fitting_algorithm.hpp"
#include "traccc/cuda/seeding/seeding_algorithm.hpp"
#include "traccc/cuda/seeding/spacepoint_formation_algorithm.hpp"
#include "traccc/cuda/seeding/track_params_estimation.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"
#include "traccc/efficiency/seeding_performance_writer.hpp"
#include "traccc/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/fitting/kalman_fitting_algorithm.hpp"
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
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_seeding.hpp"
#include "traccc/performance/collection_comparator.hpp"
#include "traccc/performance/container_comparator.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/silicon_pixel_spacepoint_formation_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// Detray include(s).
#include "detray/core/detector.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/io/frontend/detector_reader.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"

// VecMem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

// System include(s).
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>

int seq_run(const traccc::opts::detector& detector_opts,
            const traccc::opts::input_data& input_opts,
            const traccc::opts::clusterization& clusterization_opts,
            const traccc::opts::track_seeding& seeding_opts,
            const traccc::opts::track_finding& finding_opts,
            const traccc::opts::track_propagation& propagation_opts,
            const traccc::opts::performance& performance_opts,
            const traccc::opts::accelerator& accelerator_opts) {

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::host_memory_resource cuda_host_mr;
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &cuda_host_mr};

    // CUDA types used.
    traccc::cuda::stream stream;
    vecmem::cuda::async_copy copy{stream.cudaStream()};

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
        device_mr};
    copy.setup(device_det_descr)->wait();
    copy(host_det_descr_data, device_det_descr)->wait();

    // Construct a Detray detector object, if supported by the configuration.
    traccc::default_detector::host host_detector{host_mr};
    traccc::default_detector::buffer device_detector;
    traccc::default_detector::view device_detector_view;
    if (detector_opts.use_detray_detector) {
        traccc::io::read_detector(
            host_detector, host_mr, detector_opts.detector_file,
            detector_opts.material_file, detector_opts.grid_file);
        device_detector = detray::get_buffer(detray::get_data(host_detector),
                                             device_mr, copy);
        stream.synchronize();
        device_detector_view = detray::get_data(device_detector);
    }

    // Output stats
    uint64_t n_cells = 0;
    uint64_t n_measurements = 0;
    uint64_t n_measurements_cuda = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_spacepoints_cuda = 0;
    uint64_t n_seeds = 0;
    uint64_t n_seeds_cuda = 0;
    uint64_t n_found_tracks = 0;
    uint64_t n_found_tracks_cuda = 0;
    uint64_t n_fitted_tracks = 0;
    uint64_t n_fitted_tracks_cuda = 0;

    // Type definitions
    using host_spacepoint_formation_algorithm =
        traccc::host::silicon_pixel_spacepoint_formation_algorithm;
    using device_spacepoint_formation_algorithm =
        traccc::cuda::spacepoint_formation_algorithm<
            traccc::default_detector::device>;
    using stepper_type =
        detray::rk_stepper<detray::bfield::const_field_t::view_t,
                           traccc::default_detector::host::algebra_type,
                           detray::constrained_step<>>;
    using device_navigator_type =
        detray::navigator<const traccc::default_detector::device>;

    using host_finding_algorithm =
        traccc::host::combinatorial_kalman_filter_algorithm;
    using device_finding_algorithm =
        traccc::cuda::finding_algorithm<stepper_type, device_navigator_type>;

    using host_fitting_algorithm = traccc::host::kalman_fitting_algorithm;
    using device_fitting_algorithm = traccc::cuda::fitting_algorithm<
        traccc::kalman_fitter<stepper_type, device_navigator_type>>;

    // Algorithm configuration(s).
    detray::propagation::config propagation_config(propagation_opts);

    host_finding_algorithm::config_type finding_cfg(finding_opts);
    finding_cfg.propagation = propagation_config;

    host_fitting_algorithm::config_type fitting_cfg;
    fitting_cfg.propagation = propagation_config;

    // Constant B field for the track finding and fitting
    const traccc::vector3 field_vec = {0.f, 0.f,
                                       seeding_opts.seedfinder.bFieldInZ};
    const detray::bfield::const_field_t field =
        detray::bfield::create_const_field(field_vec);

    traccc::host::clusterization_algorithm ca(host_mr);
    host_spacepoint_formation_algorithm sf(host_mr);
    traccc::seeding_algorithm sa(seeding_opts.seedfinder,
                                 {seeding_opts.seedfinder},
                                 seeding_opts.seedfilter, host_mr);
    traccc::track_params_estimation tp(host_mr);
    host_finding_algorithm finding_alg(finding_cfg);
    host_fitting_algorithm fitting_alg(fitting_cfg, host_mr);

    traccc::cuda::clusterization_algorithm ca_cuda(mr, copy, stream,
                                                   clusterization_opts);
    traccc::cuda::measurement_sorting_algorithm ms_cuda(copy, stream);
    device_spacepoint_formation_algorithm sf_cuda(mr, copy, stream);
    traccc::cuda::seeding_algorithm sa_cuda(
        seeding_opts.seedfinder, {seeding_opts.seedfinder},
        seeding_opts.seedfilter, mr, copy, stream);
    traccc::cuda::track_params_estimation tp_cuda(mr, copy, stream);
    device_finding_algorithm finding_alg_cuda(finding_cfg, mr, copy, stream);
    device_fitting_algorithm fitting_alg_cuda(fitting_cfg, mr, copy, stream);

    traccc::device::container_d2h_copy_alg<
        traccc::track_candidate_container_types>
        copy_track_candidates(mr, copy);
    traccc::device::container_d2h_copy_alg<traccc::track_state_container_types>
        copy_track_states(mr, copy);

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
        host_spacepoint_formation_algorithm::output_type spacepoints_per_event;
        traccc::seeding_algorithm::output_type seeds;
        traccc::track_params_estimation::output_type params;
        host_finding_algorithm::output_type track_candidates;
        host_fitting_algorithm::output_type track_states;

        // Instantiate cuda containers/collections
        traccc::measurement_collection_types::buffer measurements_cuda_buffer(
            0, *mr.host);
        traccc::spacepoint_collection_types::buffer spacepoints_cuda_buffer(
            0, *mr.host);
        traccc::seed_collection_types::buffer seeds_cuda_buffer(0, *mr.host);
        traccc::bound_track_parameters_collection_types::buffer
            params_cuda_buffer(0, *mr.host);
        traccc::track_candidate_container_types::buffer track_candidates_buffer;
        traccc::track_state_container_types::buffer track_states_buffer;

        {
            traccc::performance::timer wall_t("Wall time", elapsedTimes);

            traccc::edm::silicon_cell_collection::host cells_per_event{host_mr};

            {
                traccc::performance::timer t("File reading  (cpu)",
                                             elapsedTimes);
                // Read the cells from the relevant event file into host memory.
                static constexpr bool DEDUPLICATE = true;
                traccc::io::read_cells(cells_per_event, event,
                                       input_opts.directory, &host_det_descr,
                                       input_opts.format, DEDUPLICATE,
                                       input_opts.use_acts_geom_source);
            }  // stop measuring file reading timer

            n_cells += cells_per_event.size();

            // Create device copy of input collections
            traccc::edm::silicon_cell_collection::buffer cells_buffer(
                static_cast<unsigned int>(cells_per_event.size()), mr.main);
            copy.setup(cells_buffer)->wait();
            copy(vecmem::get_data(cells_per_event), cells_buffer)->wait();

            // CUDA
            {
                traccc::performance::timer t("Clusterization (cuda)",
                                             elapsedTimes);
                // Reconstruct it into spacepoints on the device.
                measurements_cuda_buffer =
                    ca_cuda(cells_buffer, device_det_descr);
                ms_cuda(measurements_cuda_buffer);
                stream.synchronize();
            }  // stop measuring clusterization cuda timer

            // CPU
            if (accelerator_opts.compare_with_cpu) {
                traccc::performance::timer t("Clusterization  (cpu)",
                                             elapsedTimes);
                measurements_per_event =
                    ca(vecmem::get_data(cells_per_event), host_det_descr_data);
            }  // stop measuring clusterization cpu timer

            // Perform seeding, track finding and fitting only when using a
            // Detray geometry.
            if (detector_opts.use_detray_detector) {

                // CUDA
                {
                    traccc::performance::timer t("Spacepoint formation (cuda)",
                                                 elapsedTimes);
                    spacepoints_cuda_buffer =
                        sf_cuda(device_detector_view, measurements_cuda_buffer);
                    stream.synchronize();
                }  // stop measuring spacepoint formation cuda timer

                // CPU
                if (accelerator_opts.compare_with_cpu) {
                    traccc::performance::timer t("Spacepoint formation  (cpu)",
                                                 elapsedTimes);
                    spacepoints_per_event =
                        sf(host_detector,
                           vecmem::get_data(measurements_per_event));
                }  // stop measuring spacepoint formation cpu timer

                // CUDA
                {
                    traccc::performance::timer t("Seeding (cuda)",
                                                 elapsedTimes);
                    seeds_cuda_buffer = sa_cuda(spacepoints_cuda_buffer);
                    stream.synchronize();
                }  // stop measuring seeding cuda timer

                // CPU
                if (accelerator_opts.compare_with_cpu) {
                    traccc::performance::timer t("Seeding  (cpu)",
                                                 elapsedTimes);
                    seeds = sa(spacepoints_per_event);
                }  // stop measuring seeding cpu timer

                // CUDA
                {
                    traccc::performance::timer t("Track params (cuda)",
                                                 elapsedTimes);
                    params_cuda_buffer = tp_cuda(spacepoints_cuda_buffer,
                                                 seeds_cuda_buffer, field_vec);
                    stream.synchronize();
                }  // stop measuring track params timer

                // CPU
                if (accelerator_opts.compare_with_cpu) {
                    traccc::performance::timer t("Track params  (cpu)",
                                                 elapsedTimes);
                    params = tp(spacepoints_per_event, seeds, field_vec);
                }  // stop measuring track params cpu timer

                // CUDA
                {
                    traccc::performance::timer timer{"Track finding (cuda)",
                                                     elapsedTimes};
                    track_candidates_buffer = finding_alg_cuda(
                        device_detector_view, field, measurements_cuda_buffer,
                        params_cuda_buffer);
                }

                // CPU
                if (accelerator_opts.compare_with_cpu) {
                    traccc::performance::timer timer{"Track finding (cpu)",
                                                     elapsedTimes};
                    track_candidates =
                        finding_alg(host_detector, field,
                                    vecmem::get_data(measurements_per_event),
                                    vecmem::get_data(params));
                }

                // CUDA
                {
                    traccc::performance::timer timer{"Track fitting (cuda)",
                                                     elapsedTimes};
                    track_states_buffer = fitting_alg_cuda(
                        device_detector_view, field, track_candidates_buffer);
                }

                // CPU
                if (accelerator_opts.compare_with_cpu) {
                    traccc::performance::timer timer{"Track fitting (cpu)",
                                                     elapsedTimes};
                    track_states =
                        fitting_alg(host_detector, field,
                                    traccc::get_data(track_candidates));
                }
            }

        }  // Stop measuring wall time

        /*----------------------------------
          compare cpu and cuda result
          ----------------------------------*/

        traccc::measurement_collection_types::host measurements_per_event_cuda;
        traccc::spacepoint_collection_types::host spacepoints_per_event_cuda;
        traccc::seed_collection_types::host seeds_cuda;
        traccc::bound_track_parameters_collection_types::host params_cuda;

        copy(measurements_cuda_buffer, measurements_per_event_cuda)->wait();
        copy(spacepoints_cuda_buffer, spacepoints_per_event_cuda)->wait();
        copy(seeds_cuda_buffer, seeds_cuda)->wait();
        copy(params_cuda_buffer, params_cuda)->wait();
        auto track_candidates_cuda =
            copy_track_candidates(track_candidates_buffer);
        auto track_states_cuda = copy_track_states(track_states_buffer);
        stream.synchronize();

        if (accelerator_opts.compare_with_cpu) {

            // Show which event we are currently presenting the results for.
            std::cout << "===>>> Event " << event << " <<<===" << std::endl;

            // Compare the measurements made on the host and on the device.
            traccc::collection_comparator<traccc::measurement>
                compare_measurements{"measurements"};
            compare_measurements(vecmem::get_data(measurements_per_event),
                                 vecmem::get_data(measurements_per_event_cuda));

            // Compare the spacepoints made on the host and on the device.
            traccc::collection_comparator<traccc::spacepoint>
                compare_spacepoints{"spacepoints"};
            compare_spacepoints(vecmem::get_data(spacepoints_per_event),
                                vecmem::get_data(spacepoints_per_event_cuda));

            // Compare the seeds made on the host and on the device
            traccc::collection_comparator<traccc::seed> compare_seeds{
                "seeds", traccc::details::comparator_factory<traccc::seed>{
                             vecmem::get_data(spacepoints_per_event),
                             vecmem::get_data(spacepoints_per_event_cuda)}};
            compare_seeds(vecmem::get_data(seeds),
                          vecmem::get_data(seeds_cuda));

            // Compare the track parameters made on the host and on the device.
            traccc::collection_comparator<traccc::bound_track_parameters>
                compare_track_parameters{"track parameters"};
            compare_track_parameters(vecmem::get_data(params),
                                     vecmem::get_data(params_cuda));

            // Compare tracks found on the host and on the device.
            traccc::collection_comparator<
                traccc::track_candidate_container_types::host::header_type>
                compare_track_candidates{"track candidates (header)"};
            compare_track_candidates(
                vecmem::get_data(track_candidates.get_headers()),
                vecmem::get_data(track_candidates_cuda.get_headers()));

            unsigned int n_matches = 0;
            for (unsigned int i = 0; i < track_candidates.size(); i++) {
                auto iso = traccc::details::is_same_object(
                    track_candidates.at(i).items);

                for (unsigned int j = 0; j < track_candidates_cuda.size();
                     j++) {
                    if (iso(track_candidates_cuda.at(j).items)) {
                        n_matches++;
                        break;
                    }
                }
            }

            std::cout << "  Track candidates (item) matching rate: "
                      << 100. * static_cast<double>(n_matches) /
                             static_cast<double>(
                                 std::max(track_candidates.size(),
                                          track_candidates_cuda.size()))
                      << "%" << std::endl;

            // Compare tracks fitted on the host and on the device.
            traccc::collection_comparator<
                traccc::track_state_container_types::host::header_type>
                compare_track_states{"track states"};
            compare_track_states(
                vecmem::get_data(track_states.get_headers()),
                vecmem::get_data(track_states_cuda.get_headers()));
        }
        /// Statistics
        n_measurements += measurements_per_event.size();
        n_spacepoints += spacepoints_per_event.size();
        n_seeds += seeds.size();
        n_measurements_cuda += measurements_per_event_cuda.size();
        n_spacepoints_cuda += spacepoints_per_event_cuda.size();
        n_seeds_cuda += seeds_cuda.size();
        n_found_tracks += track_candidates.size();
        n_found_tracks_cuda += track_candidates_cuda.size();
        n_fitted_tracks += track_states.size();
        n_fitted_tracks_cuda += track_states_cuda.size();

        if (performance_opts.run) {

            // TODO: Do evt_data.fill_cca_result(...) with cuda clusters and
            // measurements
        }
    }

    if (performance_opts.run) {
        sd_performance_writer.finalize();
    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_cells << " cells" << std::endl;
    std::cout << "- created (cpu)  " << n_measurements << " measurements     "
              << std::endl;
    std::cout << "- created (cuda)  " << n_measurements_cuda
              << " measurements     " << std::endl;
    std::cout << "- created (cpu)  " << n_spacepoints << " spacepoints     "
              << std::endl;
    std::cout << "- created (cuda) " << n_spacepoints_cuda
              << " spacepoints     " << std::endl;

    std::cout << "- created  (cpu) " << n_seeds << " seeds" << std::endl;
    std::cout << "- created (cuda) " << n_seeds_cuda << " seeds" << std::endl;
    std::cout << "- found (cpu)    " << n_found_tracks << " tracks"
              << std::endl;
    std::cout << "- found (cuda)   " << n_found_tracks_cuda << " tracks"
              << std::endl;
    std::cout << "- fitted (cpu)   " << n_fitted_tracks << " tracks"
              << std::endl;
    std::cout << "- fitted (cuda)  " << n_fitted_tracks_cuda << " tracks"
              << std::endl;
    std::cout << "==>Elapsed times...\n" << elapsedTimes << std::endl;

    return 0;
}

// The main routine
//
int main(int argc, char* argv[]) {

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::clusterization clusterization_opts;
    traccc::opts::track_seeding seeding_opts;
    traccc::opts::track_finding finding_opts;
    traccc::opts::track_propagation propagation_opts;
    traccc::opts::performance performance_opts;
    traccc::opts::accelerator accelerator_opts;
    traccc::opts::program_options program_opts{
        "Full Tracking Chain Using CUDA",
        {detector_opts, input_opts, clusterization_opts, seeding_opts,
         finding_opts, propagation_opts, performance_opts, accelerator_opts},
        argc,
        argv};

    // Run the application.
    return seq_run(detector_opts, input_opts, clusterization_opts, seeding_opts,
                   finding_opts, propagation_opts, performance_opts,
                   accelerator_opts);
}
