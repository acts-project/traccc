/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "../common/make_magnetic_field.hpp"
#include "traccc/cuda/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/cuda/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/cuda/utils/make_magnetic_field.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"
#include "traccc/device/container_h2d_copy_alg.hpp"
#include "traccc/efficiency/finding_performance_writer.hpp"
#include "traccc/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_detector_description.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/options/accelerator.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/magnetic_field.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_fitting.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/truth_finding.hpp"
#include "traccc/performance/collection_comparator.hpp"
#include "traccc/performance/container_comparator.hpp"
#include "traccc/performance/soa_comparator.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"
#include "traccc/utils/propagation.hpp"
#include "traccc/utils/seed_generator.hpp"

// VecMem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

// System include(s).
#include <exception>
#include <iomanip>
#include <iostream>

using namespace traccc;

int seq_run(const traccc::opts::track_finding& finding_opts,
            const traccc::opts::track_propagation& propagation_opts,
            const traccc::opts::track_fitting& fitting_opts,
            const traccc::opts::input_data& input_opts,
            const traccc::opts::detector& detector_opts,
            const traccc::opts::magnetic_field& bfield_opts,
            const traccc::opts::performance& performance_opts,
            const traccc::opts::accelerator& accelerator_opts,
            const traccc::opts::truth_finding& truth_finding_opts,
            std::unique_ptr<const traccc::Logger> ilogger) {
    TRACCC_LOCAL_LOGGER(std::move(ilogger));

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::host_memory_resource cuda_host_mr;
    vecmem::cuda::managed_memory_resource mng_mr;
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &cuda_host_mr};

    // Performance writer
    traccc::finding_performance_writer find_performance_writer(
        traccc::finding_performance_writer::config{.truth_config =
                                                       truth_finding_opts},
        logger().clone("FindingPerformanceWriter"));
    traccc::fitting_performance_writer fit_performance_writer(
        traccc::fitting_performance_writer::config{},
        logger().clone("FittingPerformanceWriter"));

    // Output Stats
    uint64_t n_found_tracks = 0;
    uint64_t n_found_tracks_cuda = 0;
    uint64_t n_fitted_tracks = 0;
    uint64_t n_fitted_tracks_cuda = 0;

    /*****************************
     * Build a geometry
     *****************************/

    // B field value
    const auto host_field = traccc::details::make_magnetic_field(bfield_opts);
    const auto device_field = traccc::cuda::make_magnetic_field(host_field);

    // Construct a Detray detector object, if supported by the configuration.
    traccc::default_detector::host detector{mng_mr};
    assert(detector_opts.use_detray_detector == true);
    traccc::io::read_detector(detector, mng_mr, detector_opts.detector_file,
                              detector_opts.material_file,
                              detector_opts.grid_file);

    // Detector view object
    traccc::default_detector::view det_view = detray::get_data(detector);

    /*****************************
     * Do the reconstruction
     *****************************/

    // Stream object
    traccc::cuda::stream stream;

    // Copy object
    vecmem::copy host_copy;
    vecmem::cuda::async_copy async_copy{stream.cudaStream()};

    traccc::device::container_d2h_copy_alg<traccc::track_state_container_types>
        track_state_d2h{mr, async_copy, logger().clone("TrackStateD2HCopyAlg")};

    // Standard deviations for seed track parameters
    static constexpr std::array<traccc::scalar, traccc::e_bound_size> stddevs =
        {1e-4f * traccc::unit<traccc::scalar>::mm,
         1e-4f * traccc::unit<traccc::scalar>::mm,
         1e-3f,
         1e-3f,
         1e-4f / traccc::unit<traccc::scalar>::GeV,
         1e-4f * traccc::unit<traccc::scalar>::ns};

    // Propagation configuration
    detray::propagation::config propagation_config(propagation_opts);

    // Finding algorithm configuration
    traccc::finding_config cfg(finding_opts);
    cfg.propagation = propagation_config;

    // Finding algorithm object
    traccc::host::combinatorial_kalman_filter_algorithm host_finding(
        cfg, host_mr, logger().clone("HostFindingAlg"));
    traccc::cuda::combinatorial_kalman_filter_algorithm device_finding(
        cfg, mr, async_copy, stream, logger().clone("CudaFindingAlg"));

    // Fitting algorithm object
    traccc::fitting_config fit_cfg(fitting_opts);
    fit_cfg.propagation = propagation_config;

    traccc::host::kalman_fitting_algorithm host_fitting(
        fit_cfg, host_mr, host_copy, logger().clone("HostFittingAlg"));
    traccc::cuda::kalman_fitting_algorithm device_fitting(
        fit_cfg, mr, async_copy, stream, logger().clone("CudaFittingAlg"));

    traccc::performance::timing_info elapsedTimes;

    // Seed generator
    traccc::seed_generator<traccc::default_detector::host> sg(detector,
                                                              stddevs);

    // Iterate over events
    for (std::size_t event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        // Truth Track Candidates
        traccc::event_data evt_data(input_opts.directory, event, host_mr,
                                    input_opts.use_acts_geom_source, &detector,
                                    input_opts.format, false);

        traccc::edm::track_candidate_container<traccc::default_algebra>::host
            truth_track_candidates{host_mr};
        evt_data.generate_truth_candidates(truth_track_candidates, sg, host_mr,
                                           truth_finding_opts.m_pT_min);

        // Prepare truth seeds
        traccc::bound_track_parameters_collection_types::host seeds(mr.host);
        const std::size_t n_tracks = truth_track_candidates.tracks.size();
        for (std::size_t i_trk = 0; i_trk < n_tracks; i_trk++) {
            seeds.push_back(truth_track_candidates.tracks.at(i_trk).params());
        }

        traccc::bound_track_parameters_collection_types::buffer seeds_buffer{
            static_cast<unsigned int>(seeds.size()), mr.main};
        async_copy.setup(seeds_buffer)->wait();
        async_copy(vecmem::get_data(seeds), seeds_buffer,
                   vecmem::copy::type::host_to_device)
            ->wait();

        // Read measurements
        traccc::measurement_collection_types::host measurements_per_event{
            mr.host};
        traccc::io::read_measurements(
            measurements_per_event, event, input_opts.directory,
            (input_opts.use_acts_geom_source ? &detector : nullptr),
            input_opts.format);

        traccc::measurement_collection_types::buffer measurements_cuda_buffer(
            static_cast<unsigned int>(measurements_per_event.size()), mr.main);
        async_copy.setup(measurements_cuda_buffer)->wait();
        async_copy(vecmem::get_data(measurements_per_event),
                   measurements_cuda_buffer)
            ->wait();

        // Instantiate output cuda containers/collections
        traccc::edm::track_candidate_collection<traccc::default_algebra>::buffer
            track_candidates_cuda_buffer;

        {
            traccc::performance::timer t("Track finding  (cuda)", elapsedTimes);

            // Run finding
            track_candidates_cuda_buffer = device_finding(
                det_view, device_field, measurements_cuda_buffer, seeds_buffer);
        }

        traccc::edm::track_candidate_collection<traccc::default_algebra>::host
            track_candidates_cuda{host_mr};
        async_copy(track_candidates_cuda_buffer, track_candidates_cuda,
                   vecmem::copy::type::device_to_host)
            ->wait();

        // Instantiate cuda containers/collections
        traccc::track_state_container_types::buffer track_states_cuda_buffer{
            {{}, *(mr.host)}, {{}, *(mr.host), mr.host}};

        {
            traccc::performance::timer t("Track fitting  (cuda)", elapsedTimes);

            // Run fitting
            track_states_cuda_buffer = device_fitting(
                det_view, device_field,
                {track_candidates_cuda_buffer, measurements_cuda_buffer});
        }
        traccc::track_state_container_types::host track_states_cuda =
            track_state_d2h(track_states_cuda_buffer);

        // CPU containers
        traccc::host::combinatorial_kalman_filter_algorithm::output_type
            track_candidates{host_mr};
        traccc::host::kalman_fitting_algorithm::output_type track_states;

        if (accelerator_opts.compare_with_cpu) {

            {
                traccc::performance::timer t("Track finding  (cpu)",
                                             elapsedTimes);

                // Run finding
                track_candidates =
                    host_finding(detector, host_field,
                                 vecmem::get_data(measurements_per_event),
                                 vecmem::get_data(seeds));
            }

            {
                traccc::performance::timer t("Track fitting  (cpu)",
                                             elapsedTimes);

                // Run fitting
                track_states =
                    host_fitting(detector, host_field,
                                 {vecmem::get_data(track_candidates),
                                  vecmem::get_data(measurements_per_event)});
            }
        }

        if (accelerator_opts.compare_with_cpu) {

            // Show which event we are currently presenting the results for.
            TRACCC_INFO("===>>> Event " << event << " <<<===");

            // Compare the track parameters made on the host and on the device.
            traccc::soa_comparator<traccc::edm::track_candidate_collection<
                traccc::default_algebra>>
                compare_track_candidates{
                    "track candidates",
                    traccc::details::comparator_factory<
                        traccc::edm::track_candidate_collection<
                            traccc::default_algebra>::const_device::
                            const_proxy_type>{
                        vecmem::get_data(measurements_per_event),
                        vecmem::get_data(measurements_per_event)}};
            compare_track_candidates(vecmem::get_data(track_candidates),
                                     vecmem::get_data(track_candidates_cuda));
        }

        /// Statistics
        n_found_tracks += track_candidates.size();
        n_fitted_tracks += track_states.size();
        n_found_tracks_cuda += track_candidates_cuda.size();
        n_fitted_tracks_cuda += track_states_cuda.size();

        if (performance_opts.run) {
            find_performance_writer.write(
                vecmem::get_data(track_candidates_cuda),
                vecmem::get_data(measurements_per_event), evt_data);

            for (unsigned int i = 0; i < track_states_cuda.size(); i++) {
                const auto& trk_states_per_track =
                    track_states_cuda.at(i).items;

                const auto& fit_res = track_states_cuda[i].header;

                fit_performance_writer.write(trk_states_per_track, fit_res,
                                             detector, evt_data);
            }
        }
    }

    if (performance_opts.run) {
        find_performance_writer.finalize();
        fit_performance_writer.finalize();
    }

    TRACCC_INFO("==> Statistics ... ");
    TRACCC_INFO("- created (cuda) " << n_found_tracks_cuda << " found tracks");
    TRACCC_INFO("- created (cuda) " << n_fitted_tracks_cuda
                                    << " fitted tracks");
    TRACCC_INFO("- created  (cpu) " << n_found_tracks << " found tracks");
    TRACCC_INFO("- created  (cpu) " << n_fitted_tracks << " fitted tracks");
    TRACCC_INFO("==>Elapsed times... " << elapsedTimes);

    return 1;
}

// The main routine
//
int main(int argc, char* argv[]) {
    std::unique_ptr<const traccc::Logger> logger = traccc::getDefaultLogger(
        "TracccExampleTruthFindingCuda", traccc::Logging::Level::INFO);

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::magnetic_field bfield_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::track_finding finding_opts;
    traccc::opts::track_propagation propagation_opts;
    traccc::opts::track_fitting fitting_opts;
    traccc::opts::performance performance_opts;
    traccc::opts::accelerator accelerator_opts;
    traccc::opts::truth_finding truth_finding_config;
    traccc::opts::program_options program_opts{
        "Truth Track Finding Using CUDA",
        {detector_opts, bfield_opts, input_opts, finding_opts, propagation_opts,
         fitting_opts, performance_opts, accelerator_opts,
         truth_finding_config},
        argc,
        argv,
        logger->cloneWithSuffix("Options")};

    // Run the application.
    return seq_run(finding_opts, propagation_opts, fitting_opts, input_opts,
                   detector_opts, bfield_opts, performance_opts,
                   accelerator_opts, truth_finding_config, logger->clone());
}
