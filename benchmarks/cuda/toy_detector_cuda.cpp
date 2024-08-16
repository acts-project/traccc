/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/cuda/finding/finding_algorithm.hpp"
#include "traccc/cuda/fitting/fitting_algorithm.hpp"
#include "traccc/cuda/seeding/seeding_algorithm.hpp"
#include "traccc/cuda/seeding/track_params_estimation.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"
#include "traccc/io/event_map2.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/read_spacepoints.hpp"

// Local include(s).
#include "benchmarks/toy_detector_benchmark.hpp"

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
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// Google benchmark include(s).
#include <benchmark/benchmark.h>

BENCHMARK_F(ToyDetectorBenchmark, CUDA)(benchmark::State& state) {

    // Type declarations
    using rk_stepper_type =
        detray::rk_stepper<b_field_t::view_t,
                           typename detector_type::algebra_type,
                           detray::constrained_step<>>;
    using host_detector_type = detray::detector<detray::default_metadata>;
    using device_detector_type =
        detray::detector<detray::default_metadata,
                         detray::device_container_types>;
    using device_navigator_type = detray::navigator<const device_detector_type>;
    using device_fitter_type =
        traccc::kalman_fitter<rk_stepper_type, device_navigator_type>;

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::host_memory_resource cuda_host_mr;
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &cuda_host_mr};
    vecmem::cuda::managed_memory_resource mng_mr;

    // Copy and stream
    vecmem::copy host_copy;
    vecmem::cuda::copy copy;
    traccc::cuda::stream stream;
    vecmem::cuda::async_copy async_copy{stream.cudaStream()};

    // Read back detector file
    const std::string path = sim_dir;
    detray::io::detector_reader_config reader_cfg{};
    reader_cfg.add_file(path + "toy_detector_geometry.json")
        .add_file(path + "toy_detector_homogeneous_material.json")
        .add_file(path + "toy_detector_surface_grids.json");

    auto [det, names] =
        detray::io::read_detector<host_detector_type>(mng_mr, reader_cfg);

    // B field
    auto field = detray::bfield::create_const_field(B);

    // Algorithms
    traccc::cuda::seeding_algorithm sa_cuda(seeding_cfg, grid_cfg, filter_cfg,
                                            mr, async_copy, stream);
    traccc::cuda::track_params_estimation tp_cuda(mr, async_copy, stream);
    traccc::cuda::finding_algorithm<rk_stepper_type, device_navigator_type>
        device_finding(finding_cfg, mr, async_copy, stream);
    traccc::cuda::fitting_algorithm<device_fitter_type> device_fitting(
        fitting_cfg, mr, async_copy, stream);

    // Detector view object
    auto det_view = detray::get_data(det);

    // D2H copy object
    traccc::device::container_d2h_copy_alg<traccc::track_state_container_types>
        track_state_d2h{mr, copy};

    for (auto _ : state) {

        state.PauseTiming();

        for (int i = -10; i < n_events; i++) {

            int i_evt = i;

            // First 10 events are for cold run
            if (i < 0) {
                i_evt = 0;
            }
            // Measure the time after the cold run
            if (i == 0) {
                state.ResumeTiming();
            }

            auto& spacepoints_per_event = spacepoints[i_evt];
            auto& measurements_per_event = measurements[i_evt];

            // Initialize the containers
            traccc::seed_collection_types::buffer seeds_cuda_buffer(0,
                                                                    *(mr.host));
            traccc::bound_track_parameters_collection_types::buffer
                params_cuda_buffer(0, *mr.host);

            traccc::track_candidate_container_types::buffer
                track_candidates_cuda_buffer{{{}, *(mr.host)},
                                             {{}, *(mr.host), mr.host}};

            traccc::track_state_container_types::buffer
                track_states_cuda_buffer{{{}, *(mr.host)},
                                         {{}, *(mr.host), mr.host}};

            // Copy the spacepoint and module data to the device.
            traccc::spacepoint_collection_types::buffer spacepoints_cuda_buffer(
                spacepoints_per_event.size(), mr.main);
            async_copy(vecmem::get_data(spacepoints_per_event),
                       spacepoints_cuda_buffer);

            traccc::measurement_collection_types::buffer
                measurements_cuda_buffer(measurements_per_event.size(),
                                         mr.main);
            async_copy(vecmem::get_data(measurements_per_event),
                       measurements_cuda_buffer);

            // Run seeding
            seeds_cuda_buffer = sa_cuda(spacepoints_cuda_buffer);
            stream.synchronize();

            // Run track parameter estimation
            params_cuda_buffer =
                tp_cuda(spacepoints_cuda_buffer, seeds_cuda_buffer, B);
            stream.synchronize();

            // Navigation buffer
            auto navigation_buffer = detray::create_candidates_buffer(
                det,
                device_finding.get_config().navigation_buffer_size_scaler *
                    copy.get_size(seeds_cuda_buffer),
                mr.main, mr.host);

            // Run CKF track finding
            track_candidates_cuda_buffer =
                device_finding(det_view, field, navigation_buffer,
                               measurements_cuda_buffer, params_cuda_buffer);
            stream.synchronize();

            // Run track fitting
            track_states_cuda_buffer =
                device_fitting(det_view, field, navigation_buffer,
                               track_candidates_cuda_buffer);
            stream.synchronize();

            // Create a temporary buffer that will receive the device memory.
            auto size = track_states_cuda_buffer.headers.size();
            std::vector<std::size_t> capacities(size, 0);
            std::transform(track_states_cuda_buffer.items.host_ptr(),
                           track_states_cuda_buffer.items.host_ptr() + size,
                           capacities.begin(),
                           [](const auto& view) { return view.capacity(); });

            traccc::track_state_container_types::buffer
                track_states_host_buffer{{size, *(mr.host)},
                                         {capacities, *(mr.host), mr.host}};
            host_copy.setup(track_states_host_buffer.headers);
            host_copy.setup(track_states_host_buffer.items);

            // Copy the device container into this temporary host buffer.
            vecmem::copy::event_type header_event =
                copy(track_states_cuda_buffer.headers,
                     track_states_host_buffer.headers,
                     vecmem::copy::type::device_to_host);
            vecmem::copy::event_type item_event = copy(
                track_states_cuda_buffer.items, track_states_host_buffer.items,
                vecmem::copy::type::device_to_host);
        }
    }

    state.counters["event_throughput_Hz"] = benchmark::Counter(
        static_cast<double>(n_events), benchmark::Counter::kIsRate);    
}