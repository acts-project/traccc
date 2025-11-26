/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "../common/make_magnetic_field.hpp"
#include "traccc/cuda/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/cuda/utils/make_magnetic_field.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"
#include "traccc/device/container_h2d_copy_alg.hpp"
#include "traccc/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/options/accelerator.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/magnetic_field.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_fitting.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/performance/collection_comparator.hpp"
#include "traccc/performance/container_comparator.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"
#include "traccc/utils/propagation.hpp"
#include "traccc/utils/seed_generator.hpp"

// Detray include(s).
#include <detray/core/detail/alignment.hpp>
#include <detray/io/frontend/detector_reader.hpp>

// VecMem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// System include(s).
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>

using namespace traccc;

// The main routine
//
int main(int argc, char* argv[]) {
    std::unique_ptr<const traccc::Logger> ilogger =
        traccc::getDefaultLogger("TracccExampleMisalignedTruthFittingCuda",
                                 traccc::Logging::Level::INFO);
    TRACCC_LOCAL_LOGGER(std::move(ilogger));

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::magnetic_field bfield_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::track_propagation propagation_opts;
    traccc::opts::track_fitting fitting_opts;
    traccc::opts::performance performance_opts;
    traccc::opts::accelerator accelerator_opts;
    traccc::opts::program_options program_opts{
        "Misaligned Truth Track Fitting Using CUDA",
        {detector_opts, bfield_opts, input_opts, propagation_opts,
         performance_opts, accelerator_opts},
        argc,
        argv,
        logger().cloneWithSuffix("Options")};

    /// Type declarations
    using host_detector_type = traccc::default_detector::host;
    /*
    using device_detector_type = traccc::default_detector::device;

    using scalar_type = device_detector_type::scalar_type;
    using b_field_t =
        covfie::field<traccc::const_bfield_backend_t<scalar_type>>;
    using rk_stepper_type =
        detray::rk_stepper<b_field_t::view_t, traccc::default_algebra,
                           detray::constrained_step<scalar_type>>;
    using device_navigator_type = detray::navigator<const device_detector_type>;
    using device_fitter_type =
        traccc::kalman_fitter<rk_stepper_type, device_navigator_type>;
    */
    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::host_memory_resource cuda_host_mr;
    vecmem::cuda::managed_memory_resource mng_mr;
    vecmem::cuda::device_memory_resource device_mr;
    vecmem::cuda::copy cuda_cpy;
    traccc::memory_resource mr{device_mr, &cuda_host_mr};

    // Performance writer
    traccc::fitting_performance_writer fit_performance_writer(
        traccc::fitting_performance_writer::config{},
        logger().clone("FittingPerformanceWriter"));

    // Output Stats
    std::size_t n_fitted_tracks = 0;
    std::size_t n_fitted_tracks_cuda = 0;

    /*****************************
     * Build a geometry
     *****************************/

    // B field value
    const auto host_field = traccc::details::make_magnetic_field(bfield_opts);
    const auto device_field = traccc::cuda::make_magnetic_field(host_field);

    // Read the detector
    detray::io::detector_reader_config reader_cfg{};
    reader_cfg.add_file(
        traccc::io::get_absolute_path(detector_opts.detector_file));
    if (!detector_opts.material_file.empty()) {
        reader_cfg.add_file(
            traccc::io::get_absolute_path(detector_opts.material_file));
    }
    if (!detector_opts.grid_file.empty()) {
        reader_cfg.add_file(
            traccc::io::get_absolute_path(detector_opts.grid_file));
    }
    auto [host_det, names] =
        detray::io::read_detector<host_detector_type>(host_mr, reader_cfg);

    // Copy detector to the device
    auto det_buff_static = detray::get_buffer(host_det, device_mr, cuda_cpy);

    // Detector view object
    auto det_view_static = detray::get_data(det_buff_static);

    /// Create a context of misaligned transforms for running on host and device
    /// Step 1. On the host, create an instance of the transform store
    /// with the same size as the transform store of the detector object,
    /// and then fill it with misaligned transforms
    using xf_container = host_detector_type::transform_container;
    xf_container tf_store_aligned_host;
    tf_store_aligned_host.reserve(
        host_det.transform_store().size(),
        typename host_detector_type::transform_container::context_type{});
    for (const auto& tf : host_det.transform_store()) {
        tf_store_aligned_host.push_back(tf);
    }

    /// Step 2. Get the vector of transformations from the newly created
    /// transform store and insert it into the detector transform store as a
    /// misaligned context
    using xf_vector = xf_container::base_type;
    xf_vector* tf_aligned_host = tf_store_aligned_host.data();
    const xf_container& default_xfs = host_det.transform_store();
    xf_container* ptr_default_xfs = const_cast<xf_container*>(&default_xfs);
    ptr_default_xfs->add_context(*tf_aligned_host);

    /// Step 3. Copy the vector of misaligned transforms to the device
    auto tf_buff_aligned = detray::get_buffer(
        tf_store_aligned_host, device_mr, cuda_cpy, detray::copy::sync,
        vecmem::data::buffer_type::fixed_size);

    /// Step 4. Get the view of the misaligned detector using the vector of
    /// misaligned transforms and the static part of the detector copied to the
    /// device earlier
    auto det_view_aligned =
        detray::detail::misaligned_detector_view<host_detector_type>(
            det_buff_static, tf_buff_aligned);

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

    /// Standard deviations for seed track parameters
    static constexpr std::array<scalar, e_bound_size> stddevs = {
        0.03f * traccc::unit<scalar>::mm,
        0.03f * traccc::unit<scalar>::mm,
        0.017f,
        0.017f,
        0.001f / traccc::unit<scalar>::GeV,
        1.f * traccc::unit<scalar>::ns};

    // Fitting algorithm objects
    traccc::fitting_config fit_cfg0(fitting_opts);
    fit_cfg0.propagation = propagation_opts;
    fit_cfg0.propagation.context = host_detector_type::geometry_context{0};
    traccc::host::kalman_fitting_algorithm host_fitting0(
        fit_cfg0, host_mr, host_copy, logger().clone("HostFittingAlg0"));

    traccc::fitting_config fit_cfg1(fitting_opts);
    fit_cfg1.propagation = propagation_opts;
    fit_cfg1.propagation.context = host_detector_type::geometry_context{1};
    traccc::host::kalman_fitting_algorithm host_fitting1(
        fit_cfg1, host_mr, host_copy, logger().clone("HostFittingAlg1"));

    // Do we alson need two instances of device fitting algorithms?
    traccc::cuda::kalman_fitting_algorithm device_fitting(
        fit_cfg0, mr, async_copy, stream, logger().clone("CudaFittingAlg"));

    // Seed generators
    traccc::seed_generator<host_detector_type> sg0(
        host_det, stddevs, 0, fit_cfg0.propagation.context);
    traccc::seed_generator<host_detector_type> sg1(
        host_det, stddevs, 0, fit_cfg1.propagation.context);

    traccc::performance::timing_info elapsedTimes;

    // Iterate over events
    for (std::size_t event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        // Truth Track Candidates
        traccc::event_data evt_data(input_opts.directory, event, host_mr,
                                    input_opts.use_acts_geom_source, &host_det,
                                    input_opts.format, false);

        traccc::edm::track_candidate_container<traccc::default_algebra>::host
            truth_track_candidates{host_mr};

        // Context 0: first half of the events
        if ((event - input_opts.skip) / (input_opts.events / 2) == 0) {
            evt_data.generate_truth_candidates(truth_track_candidates, sg0,
                                               host_mr);

            // track candidates buffer
            traccc::edm::track_candidate_container<
                traccc::default_algebra>::buffer truth_track_candidates_buffer{
                async_copy.to(vecmem::get_data(truth_track_candidates.tracks),
                              mr.main, mr.host,
                              vecmem::copy::type::host_to_device),
                async_copy.to(
                    vecmem::get_data(truth_track_candidates.measurements),
                    mr.main, vecmem::copy::type::host_to_device)};

            // Instantiate cuda containers/collections
            traccc::track_state_container_types::buffer
                track_states_cuda_buffer{{{}, *(mr.host)},
                                         {{}, *(mr.host), mr.host}};

            // Run fitting on the device
            {
                traccc::performance::timer t("Track fitting  (cuda)",
                                             elapsedTimes);

                track_states_cuda_buffer = device_fitting(
                    det_view_static, device_field,
                    {truth_track_candidates_buffer.tracks,
                     truth_track_candidates_buffer.measurements});
            }

            traccc::track_state_container_types::host track_states_cuda =
                track_state_d2h(track_states_cuda_buffer);

            // CPU container(s)
            traccc::host::kalman_fitting_algorithm::output_type track_states;

            if (accelerator_opts.compare_with_cpu) {

                {
                    traccc::performance::timer t("Track fitting  (cpu)",
                                                 elapsedTimes);

                    // Run fitting
                    track_states = host_fitting0(
                        host_det, host_field,
                        {vecmem::get_data(truth_track_candidates.tracks),
                         vecmem::get_data(
                             truth_track_candidates.measurements)});
                }

                // Show which event we are currently presenting the results for.
                std::cout << "===>>> Event " << event << " <<<===" << std::endl;

                // Compare the track parameters made on the host and on the
                // device.
                traccc::collection_comparator<
                    traccc::fitting_result<traccc::default_algebra>>
                    compare_fitting_results{"fitted tracks"};
                compare_fitting_results(
                    vecmem::get_data(track_states.get_headers()),
                    vecmem::get_data(track_states_cuda.get_headers()));
            }

            // Statistics
            n_fitted_tracks += track_states.size();
            n_fitted_tracks_cuda += track_states_cuda.size();

            if (performance_opts.run) {
                for (unsigned int i = 0; i < track_states_cuda.size(); i++) {
                    const auto& trk_states_per_track =
                        track_states_cuda.at(i).items;

                    const auto& fit_res = track_states_cuda[i].header;

                    fit_performance_writer.write(trk_states_per_track, fit_res,
                                                 host_det, evt_data);
                }
            }
        } else {
            // Context 1: second half of the events
            evt_data.generate_truth_candidates(truth_track_candidates, sg1,
                                               host_mr);

            // track candidates buffer
            traccc::edm::track_candidate_container<
                traccc::default_algebra>::buffer truth_track_candidates_buffer{
                async_copy.to(vecmem::get_data(truth_track_candidates.tracks),
                              mr.main, mr.host,
                              vecmem::copy::type::host_to_device),
                async_copy.to(
                    vecmem::get_data(truth_track_candidates.measurements),
                    mr.main, vecmem::copy::type::host_to_device)};

            // Instantiate cuda containers/collections
            traccc::track_state_container_types::buffer
                track_states_cuda_buffer{{{}, *(mr.host)},
                                         {{}, *(mr.host), mr.host}};

            // Run fitting on the device
            {
                traccc::performance::timer t("Track fitting  (cuda)",
                                             elapsedTimes);

                track_states_cuda_buffer = device_fitting(
                    det_view_aligned, device_field,
                    {truth_track_candidates_buffer.tracks,
                     truth_track_candidates_buffer.measurements});
            }

            traccc::track_state_container_types::host track_states_cuda =
                track_state_d2h(track_states_cuda_buffer);

            // CPU container(s)
            traccc::host::kalman_fitting_algorithm::output_type track_states;

            if (accelerator_opts.compare_with_cpu) {

                {
                    traccc::performance::timer t("Track fitting  (cpu)",
                                                 elapsedTimes);

                    // Run fitting
                    track_states = host_fitting1(
                        host_det, host_field,
                        {vecmem::get_data(truth_track_candidates.tracks),
                         vecmem::get_data(
                             truth_track_candidates.measurements)});
                }

                // Show which event we are currently presenting the results for.
                std::cout << "===>>> Event " << event << " <<<===" << std::endl;

                // Compare the track parameters made on the host and on the
                // device.
                traccc::collection_comparator<
                    traccc::fitting_result<traccc::default_algebra>>
                    compare_fitting_results{"fitted tracks"};
                compare_fitting_results(
                    vecmem::get_data(track_states.get_headers()),
                    vecmem::get_data(track_states_cuda.get_headers()));
            }

            // Statistics
            n_fitted_tracks += track_states.size();
            n_fitted_tracks_cuda += track_states_cuda.size();

            if (performance_opts.run) {
                for (unsigned int i = 0; i < track_states_cuda.size(); i++) {
                    const auto& trk_states_per_track =
                        track_states_cuda.at(i).items;

                    const auto& fit_res = track_states_cuda[i].header;

                    fit_performance_writer.write(trk_states_per_track, fit_res,
                                                 host_det, evt_data);
                }
            }
        }
    }

    if (performance_opts.run) {
        fit_performance_writer.finalize();
    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- created (cuda) " << n_fitted_tracks_cuda << " fitted tracks"
              << std::endl;
    std::cout << "- created  (cpu) " << n_fitted_tracks << " fitted tracks"
              << std::endl;
    std::cout << "==>Elapsed times...\n" << elapsedTimes << std::endl;

    return EXIT_SUCCESS;
}
