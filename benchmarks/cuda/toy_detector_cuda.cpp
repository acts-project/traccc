/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/bfield/construct_const_bfield.hpp"
#include "traccc/cuda/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/cuda/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/cuda/seeding/seeding_algorithm.hpp"
#include "traccc/cuda/seeding/track_params_estimation.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/read_spacepoints.hpp"
#include "traccc/utils/propagation.hpp"

// Local include(s).
#include "benchmarks/toy_detector_benchmark.hpp"

// Detray include(s).
#include <detray/io/frontend/detector_reader.hpp>

// VecMem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// Google benchmark include(s).
#include <benchmark/benchmark.h>

BENCHMARK_DEFINE_F(ToyDetectorBenchmark, CUDA)(benchmark::State& state) {

    // Memory resources used by the application.
    vecmem::cuda::host_memory_resource cuda_host_mr;
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &cuda_host_mr};

    // Copy and stream
    vecmem::copy host_copy;
    vecmem::cuda::copy copy;
    traccc::cuda::stream stream;
    vecmem::cuda::async_copy async_copy{stream.cudaStream()};

    // Read back detector file
    traccc::host_detector det;
    traccc::io::read_detector(
        det, cuda_host_mr, sim_dir + "toy_detector_geometry.json",
        sim_dir + "toy_detector_homogeneous_material.json",
        sim_dir + "toy_detector_surface_grids.json");

    // B field
    const auto field = traccc::construct_const_bfield(B);

    // Algorithms
    traccc::cuda::seeding_algorithm sa_cuda(seeding_cfg, grid_cfg, filter_cfg,
                                            mr, async_copy, stream);
    traccc::track_params_estimation_config track_params_estimation_config;
    traccc::cuda::track_params_estimation tp_cuda(
        track_params_estimation_config, mr, async_copy, stream);
    traccc::cuda::combinatorial_kalman_filter_algorithm device_finding(
        finding_cfg, mr, async_copy, stream);
    traccc::cuda::kalman_fitting_algorithm device_fitting(fitting_cfg, mr,
                                                          async_copy, stream);

    // Copy detector to device
    const traccc::detector_buffer det_buffer =
        traccc::buffer_from_host_detector(det, device_mr, copy);

    for (auto _ : state) {

        state.PauseTiming();

        for (int i = -10; i < n_events; i++) {

            // First 10 events are for cold run
            auto i_evt = static_cast<unsigned int>(std::max(i, 0));

            // Measure the time after the cold run
            if (i == 0) {
                state.ResumeTiming();
            }

            auto& spacepoints_per_event = spacepoints[i_evt];
            auto& measurements_per_event = measurements[i_evt];

            // Copy the spacepoint and module data to the device.
            traccc::edm::spacepoint_collection::buffer spacepoints_cuda_buffer(
                static_cast<unsigned int>(spacepoints_per_event.size()),
                mr.main);
            async_copy.setup(spacepoints_cuda_buffer)->ignore();
            async_copy(vecmem::get_data(spacepoints_per_event),
                       spacepoints_cuda_buffer)
                ->ignore();

            traccc::measurement_collection_types::buffer
                measurements_cuda_buffer(
                    static_cast<unsigned int>(measurements_per_event.size()),
                    mr.main);
            async_copy.setup(measurements_cuda_buffer)->ignore();
            async_copy(vecmem::get_data(measurements_per_event),
                       measurements_cuda_buffer)
                ->ignore();

            // Run seeding
            traccc::edm::seed_collection::buffer seeds_cuda_buffer =
                sa_cuda(spacepoints_cuda_buffer);

            // Run track parameter estimation
            traccc::bound_track_parameters_collection_types::buffer
                params_cuda_buffer =
                    tp_cuda(measurements_cuda_buffer, spacepoints_cuda_buffer,
                            seeds_cuda_buffer, B);

            // Run CKF track finding
            traccc::edm::track_candidate_collection<
                traccc::default_algebra>::buffer track_candidates_cuda_buffer =
                device_finding(det_buffer, field, measurements_cuda_buffer,
                               params_cuda_buffer);

            // Run track fitting
            traccc::edm::track_fit_container<traccc::default_algebra>::buffer
                track_states_cuda_buffer = device_fitting(
                    det_buffer, field,
                    {track_candidates_cuda_buffer, measurements_cuda_buffer});

            // Create a temporary buffer that will receive the device memory.
            /*auto size = track_states_cuda_buffer.headers.size();
            std::vector<std::size_t> capacities(size, 0);
            std::transform(track_states_cuda_buffer.items.host_ptr(),
                           track_states_cuda_buffer.items.host_ptr() + size,
                           capacities.begin(),
                           [](const auto& view) { return view.capacity(); });

            // Copy the track states back to the host.
            traccc::track_state_container_types::host track_states_host =
                track_state_d2h(track_states_cuda_buffer);*/
        }
    }

    state.counters["event_throughput_Hz"] = benchmark::Counter(
        static_cast<double>(n_events), benchmark::Counter::kIsRate);
}

BENCHMARK_REGISTER_F(ToyDetectorBenchmark, CUDA)->UseRealTime();
