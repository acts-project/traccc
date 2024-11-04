/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/cuda/finding/finding_algorithm.hpp"
#include "traccc/cuda/fitting/fitting_algorithm.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"
#include "traccc/device/container_h2d_copy_alg.hpp"
#include "traccc/efficiency/finding_performance_writer.hpp"
#include "traccc/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_detector_description.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/options/accelerator.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/performance/collection_comparator.hpp"
#include "traccc/performance/container_comparator.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"
#include "traccc/utils/seed_generator.hpp"

// detray include(s).
#include "detray/core/detector.hpp"
#include "detray/core/detector_metadata.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/io/frontend/detector_reader.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"

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
            const traccc::opts::input_data& input_opts,
            const traccc::opts::detector& detector_opts,
            const traccc::opts::performance& performance_opts,
            const traccc::opts::accelerator& accelerator_opts) {

    /// Type declarations
    using b_field_t = covfie::field<detray::bfield::const_bknd_t>;
    using rk_stepper_type =
        detray::rk_stepper<b_field_t::view_t, traccc::default_algebra,
                           detray::constrained_step<>>;
    using device_navigator_type =
        detray::navigator<const traccc::default_detector::device>;
    using device_fitter_type =
        traccc::kalman_fitter<rk_stepper_type, device_navigator_type>;

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::host_memory_resource cuda_host_mr;
    vecmem::cuda::managed_memory_resource mng_mr;
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &cuda_host_mr};

    // Performance writer
    traccc::finding_performance_writer find_performance_writer(
        traccc::finding_performance_writer::config{});
    traccc::fitting_performance_writer fit_performance_writer(
        traccc::fitting_performance_writer::config{});

    // Output Stats
    uint64_t n_found_tracks = 0;
    uint64_t n_found_tracks_cuda = 0;
    uint64_t n_fitted_tracks = 0;
    uint64_t n_fitted_tracks_cuda = 0;

    /*****************************
     * Build a geometry
     *****************************/

    // B field value and its type
    // @TODO: Set B field as argument
    const traccc::vector3 B{0, 0, 2 * detray::unit<traccc::scalar>::T};
    auto field = detray::bfield::create_const_field(B);

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
    vecmem::cuda::async_copy async_copy{stream.cudaStream()};

    traccc::device::container_d2h_copy_alg<
        traccc::track_candidate_container_types>
        track_candidate_d2h{mr, async_copy};

    traccc::device::container_d2h_copy_alg<traccc::track_state_container_types>
        track_state_d2h{mr, async_copy};

    // Standard deviations for seed track parameters
    static constexpr std::array<traccc::scalar, traccc::e_bound_size> stddevs =
        {1e-4f * detray::unit<traccc::scalar>::mm,
         1e-4f * detray::unit<traccc::scalar>::mm,
         1e-3f,
         1e-3f,
         1e-4f / detray::unit<traccc::scalar>::GeV,
         1e-4f * detray::unit<traccc::scalar>::ns};

    // Propagation configuration
    detray::propagation::config propagation_config(propagation_opts);

    // Finding algorithm configuration
    typename traccc::cuda::finding_algorithm<
        rk_stepper_type, device_navigator_type>::config_type cfg(finding_opts);
    cfg.propagation = propagation_config;

    // Finding algorithm object
    traccc::host::combinatorial_kalman_filter_algorithm host_finding(cfg);
    traccc::cuda::finding_algorithm<rk_stepper_type, device_navigator_type>
        device_finding(cfg, mr, async_copy, stream);

    // Fitting algorithm object
    traccc::fitting_config fit_cfg;
    fit_cfg.propagation = propagation_config;

    traccc::host::kalman_fitting_algorithm host_fitting(fit_cfg, host_mr);
    traccc::cuda::fitting_algorithm<device_fitter_type> device_fitting(
        fit_cfg, mr, async_copy, stream);

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

        traccc::track_candidate_container_types::host truth_track_candidates =
            evt_data.generate_truth_candidates(sg, host_mr);

        // Prepare truth seeds
        traccc::bound_track_parameters_collection_types::host seeds(mr.host);
        const std::size_t n_tracks = truth_track_candidates.size();
        for (std::size_t i_trk = 0; i_trk < n_tracks; i_trk++) {
            seeds.push_back(truth_track_candidates.at(i_trk).header);
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
        traccc::track_candidate_container_types::buffer
            track_candidates_cuda_buffer{{{}, *(mr.host)},
                                         {{}, *(mr.host), mr.host}};
        async_copy.setup(track_candidates_cuda_buffer.headers)->wait();
        async_copy.setup(track_candidates_cuda_buffer.items)->wait();

        {
            traccc::performance::timer t("Track finding  (cuda)", elapsedTimes);

            // Run finding
            track_candidates_cuda_buffer = device_finding(
                det_view, field, measurements_cuda_buffer, seeds_buffer);
        }

        traccc::track_candidate_container_types::host track_candidates_cuda =
            track_candidate_d2h(track_candidates_cuda_buffer);

        // Instantiate cuda containers/collections
        traccc::track_state_container_types::buffer track_states_cuda_buffer{
            {{}, *(mr.host)}, {{}, *(mr.host), mr.host}};

        {
            traccc::performance::timer t("Track fitting  (cuda)", elapsedTimes);

            // Run fitting
            track_states_cuda_buffer =
                device_fitting(det_view, field, track_candidates_cuda_buffer);
        }
        traccc::track_state_container_types::host track_states_cuda =
            track_state_d2h(track_states_cuda_buffer);

        // CPU containers
        traccc::host::combinatorial_kalman_filter_algorithm::output_type
            track_candidates;
        traccc::host::kalman_fitting_algorithm::output_type track_states;

        if (accelerator_opts.compare_with_cpu) {

            {
                traccc::performance::timer t("Track finding  (cpu)",
                                             elapsedTimes);

                // Run finding
                track_candidates = host_finding(
                    detector, field, vecmem::get_data(measurements_per_event),
                    vecmem::get_data(seeds));
            }

            {
                traccc::performance::timer t("Track fitting  (cpu)",
                                             elapsedTimes);

                // Run fitting
                track_states = host_fitting(detector, field,
                                            traccc::get_data(track_candidates));
            }
        }

        if (accelerator_opts.compare_with_cpu) {

            // Show which event we are currently presenting the results for.
            std::cout << "===>>> Event " << event << " <<<===" << std::endl;
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
            std::cout << "Track candidate matching Rate: "
                      << float(n_matches) /
                             static_cast<float>(track_candidates.size())
                      << std::endl;

            // Compare the track parameters made on the host and on the device.
            traccc::collection_comparator<
                traccc::fitting_result<traccc::default_algebra>>
                compare_fitting_results{"fitted tracks"};
            compare_fitting_results(
                vecmem::get_data(track_states.get_headers()),
                vecmem::get_data(track_states_cuda.get_headers()));
        }

        /// Statistics
        n_found_tracks += track_candidates.size();
        n_fitted_tracks += track_states.size();
        n_found_tracks_cuda += track_candidates_cuda.size();
        n_fitted_tracks_cuda += track_states_cuda.size();

        if (performance_opts.run) {
            find_performance_writer.write(
                traccc::get_data(track_candidates_cuda), evt_data);

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

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- created (cuda) " << n_found_tracks_cuda << " found tracks"
              << std::endl;
    std::cout << "- created (cuda) " << n_fitted_tracks_cuda << " fitted tracks"
              << std::endl;
    std::cout << "- created  (cpu) " << n_found_tracks << " found tracks"
              << std::endl;
    std::cout << "- created  (cpu) " << n_fitted_tracks << " fitted tracks"
              << std::endl;
    std::cout << "==>Elapsed times...\n" << elapsedTimes << std::endl;

    return 1;
}

// The main routine
//
int main(int argc, char* argv[]) {

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::track_finding finding_opts;
    traccc::opts::track_propagation propagation_opts;
    traccc::opts::performance performance_opts;
    traccc::opts::accelerator accelerator_opts;
    traccc::opts::program_options program_opts{
        "Truth Track Finding Using CUDA",
        {detector_opts, input_opts, finding_opts, propagation_opts,
         performance_opts, accelerator_opts},
        argc,
        argv};

    // Run the application.
    return seq_run(finding_opts, propagation_opts, input_opts, detector_opts,
                   performance_opts, accelerator_opts);
}
