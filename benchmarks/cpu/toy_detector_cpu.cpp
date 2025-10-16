/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Traccc core include(s).
#include "traccc/bfield/construct_const_bfield.hpp"
#include "traccc/geometry/detector.hpp"

// Traccc algorithm include(s).
#include "traccc/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// Traccc IO include(s).
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/read_spacepoints.hpp"

// Local include(s).
#include "benchmarks/toy_detector_benchmark.hpp"

// Detray include(s).
#include <detray/io/frontend/detector_reader.hpp>

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// Google benchmark include(s).
#include <benchmark/benchmark.h>

BENCHMARK_DEFINE_F(ToyDetectorBenchmark, CPU)(benchmark::State& state) {

    // VecMem copy object
    vecmem::copy copy;

    // Read back detector file
    traccc::host_detector det;
    traccc::io::read_detector(
        det, host_mr, sim_dir + "toy_detector_geometry.json",
        sim_dir + "toy_detector_homogeneous_material.json",
        sim_dir + "toy_detector_surface_grids.json");

    // B field
    const auto field = traccc::construct_const_bfield(B);

    // Algorithms
    traccc::host::seeding_algorithm sa(seeding_cfg, grid_cfg, filter_cfg,
                                       host_mr);
    traccc::track_params_estimation_config track_params_estimation_config;
    traccc::host::track_params_estimation tp(track_params_estimation_config,
                                             host_mr);
    traccc::host::combinatorial_kalman_filter_algorithm host_finding(
        finding_cfg, host_mr);
    traccc::host::kalman_fitting_algorithm host_fitting(fitting_cfg, host_mr,
                                                        copy);

    for (auto _ : state) {

// Iterate over events
#pragma omp parallel for schedule(dynamic)
        for (unsigned int i_evt = 0; i_evt < n_events; i_evt++) {

            auto& spacepoints_per_event = spacepoints[i_evt];
            auto& measurements_per_event = measurements[i_evt];

            // Seeding
            auto seeds = sa(vecmem::get_data(spacepoints_per_event));

            // Track param estimation
            auto params = tp(vecmem::get_data(measurements_per_event),
                             vecmem::get_data(spacepoints_per_event),
                             vecmem::get_data(seeds), B);

            // Track finding with CKF
            auto track_candidates = host_finding(
                det, field, vecmem::get_data(measurements_per_event),
                vecmem::get_data(params));

            // Track fitting with KF
            auto track_states =
                host_fitting(det, field,
                             {vecmem::get_data(track_candidates),
                              vecmem::get_data(measurements_per_event)});
        }
    }

    state.counters["event_throughput_Hz"] = benchmark::Counter(
        static_cast<double>(n_events), benchmark::Counter::kIsRate);
}

BENCHMARK_REGISTER_F(ToyDetectorBenchmark, CPU)->UseRealTime();
