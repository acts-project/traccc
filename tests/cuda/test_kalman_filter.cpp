/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "tests/seed_generator.hpp"
#include "traccc/cuda/fitting/fitting_algorithm.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"
#include "traccc/device/container_h2d_copy_alg.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
#include "traccc/performance/details/is_same_object.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"
#include "traccc/utils/memory_resource.hpp"

// Test include(s).
#include "tests/kalman_fitting_test.hpp"

// detray include(s).
#include "detray/detectors/create_telescope_detector.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/simulation/event_generator/track_generators.hpp"
#include "detray/simulation/simulator.hpp"

// VecMem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <climits>
#include <iostream>

using namespace traccc;

// This defines the local frame test suite
TEST_P(KalmanFittingTests, Run) {

    const std::string dir = std::get<0>(GetParam());
    const unsigned int n_truth_tracks = std::get<1>(GetParam());
    const unsigned int n_events = std::get<2>(GetParam());

    // Input path
    const std::string full_path =
        "detray_simulation/telescope/kf_validation/" + dir + "/";

    // Performance writer
    traccc::fitting_performance_writer::config writer_cfg;
    writer_cfg.file_path = "performance_track_fitting_" + dir + ".root";

    traccc::fitting_performance_writer fit_performance_writer(writer_cfg);

    /*****************************
     * Build a telescope geometry
     *****************************/

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &host_mr};
    vecmem::cuda::managed_memory_resource mng_mr;

    // Use rectangle surfaces
    detray::mask<detray::unbounded<detray::rectangle2D<>>> rectangle{
        0u, 10000.f * detray::unit<scalar>::mm,
        10000.f * detray::unit<scalar>::mm};

    host_detector_type host_det = create_telescope_detector(
        mng_mr,
        b_field_t(b_field_t::backend_t::configuration_t{B[0], B[1], B[2]}),
        rectangle, plane_positions, mat, thickness, traj);

    /***************
     * Run fitting
     ***************/

    vecmem::cuda::copy copy;

    traccc::device::container_h2d_copy_alg<
        traccc::track_candidate_container_types>
        track_candidate_h2d{mr, copy};

    traccc::device::container_d2h_copy_alg<traccc::track_state_container_types>
        track_state_d2h{mr, copy};

    // Seed generator
    seed_generator<rk_stepper_type, host_navigator_type> sg(host_det, stddevs);

    // Fitting algorithm object
    traccc::cuda::fitting_algorithm<device_fitter_type> device_fitting(mr);

    // Iterate over events
    for (std::size_t i_evt = 0; i_evt < n_events; i_evt++) {
        // Event map
        traccc::event_map2 evt_map(i_evt, full_path, full_path, full_path);

        // Truth Track Candidates
        traccc::track_candidate_container_types::host track_candidates =
            evt_map.generate_truth_candidates(sg, mng_mr);

        // Instantiate cuda containers/collections
        traccc::track_state_container_types::buffer track_states_cuda_buffer{
            {{}, *(mr.host)}, {{}, *(mr.host), mr.host}};

        // n_trakcs = 100
        ASSERT_EQ(track_candidates.size(), n_truth_tracks);

        // Detector view object
        auto det_view = detray::get_data(host_det);

        // Navigation buffer
        auto navigation_buffer = detray::create_candidates_buffer(
            host_det, track_candidates.size(), mr.main, mr.host);

        // track candidates buffer
        const traccc::track_candidate_container_types::buffer
            track_candidates_cuda_buffer =
                track_candidate_h2d(traccc::get_data(track_candidates));

        // Run fitting
        track_states_cuda_buffer = device_fitting(det_view, navigation_buffer,
                                                  track_candidates_cuda_buffer);

        traccc::track_state_container_types::host track_states_cuda =
            track_state_d2h(track_states_cuda_buffer);

        ASSERT_EQ(track_states_cuda.size(), n_truth_tracks);

        const std::size_t n_tracks = track_states_cuda.size();

        for (std::size_t i_trk = 0; i_trk < n_tracks; i_trk++) {
            auto& device_states = track_states_cuda[i_trk].items;

            fit_performance_writer.write(device_states, host_det, evt_map);
        }
    }

    fit_performance_writer.finalize();

    /********************
     * Pull value test
     ********************/

    static const std::vector<std::string> pull_names{
        "pull_d0", "pull_z0", "pull_phi", "pull_theta", "pull_qop"};
    pull_value_tests(writer_cfg.file_path, pull_names);
}

INSTANTIATE_TEST_SUITE_P(
    KalmanFitValidation, KalmanFittingTests,
    ::testing::Values(std::make_tuple("1_GeV_0_phi", 100, 100),
                      std::make_tuple("10_GeV_0_phi", 100, 100),
                      std::make_tuple("100_GeV_0_phi", 100, 100)));
