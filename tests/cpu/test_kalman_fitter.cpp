/** TRACCC library, part of the ACTS project (R&D line)
- *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "tests/seed_generator.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"

// Test include(s).
#include "tests/kalman_fitting_test.hpp"

// detray include(s).
#include "detray/detectors/create_telescope_detector.hpp"
#include "detray/simulation/track_generators.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <climits>

using namespace traccc;

// This defines the local frame test suite
TEST_P(KalmanFittingTests, Run) {

    // Test Parameters
    const scalar p0 = std::get<0>(GetParam());
    const scalar phi0 = std::get<1>(GetParam());

    // Input path
    const std::string full_path = "detray_simulation/telescope/kf_validation/" +
                                  std::to_string(p0) + "_GeV_" +
                                  std::to_string(phi0) + "_phi/";

    // Performance writer
    traccc::fitting_performance_writer::config writer_cfg;
    writer_cfg.file_path = "performance_track_fitting_" + std::to_string(p0) +
                           "_GeV_" + std::to_string(phi0) + "_phi" + ".root";

    traccc::fitting_performance_writer fit_performance_writer(writer_cfg);

    /*****************************
     * Build a telescope geometry
     *****************************/

    // Memory resource
    vecmem::host_memory_resource host_mr;

    const host_detector_type det = create_telescope_detector(
        host_mr,
        b_field_t(b_field_t::backend_t::configuration_t{B[0], B[1], B[2]}),
        plane_positions, traj, std::numeric_limits<scalar>::infinity(),
        std::numeric_limits<scalar>::infinity(), mat, thickness);

    /***************
     * Run fitting
     ***************/

    // Seed generator
    seed_generator<rk_stepper_type, host_navigator_type> sg(det, stddevs);

    // Fitting algorithm object
    fitting_algorithm<host_fitter_type> fitting;

    std::size_t n_events = 100;

    // Iterate over events
    for (std::size_t i_evt = 0; i_evt < n_events; i_evt++) {
        // Event map
        traccc::event_map2 evt_map(i_evt, full_path, full_path, full_path);

        // Truth Track Candidates
        traccc::track_candidate_container_types::host track_candidates =
            evt_map.generate_truth_candidates(sg, host_mr);

        // n_trakcs = 100
        ASSERT_EQ(track_candidates.size(), 100);

        // Run fitting
        auto track_states = fitting(det, track_candidates);

        // Iterator over tracks
        const std::size_t n_tracks = track_states.size();

        // n_trakcs = 100
        ASSERT_EQ(n_tracks, 100);
        for (std::size_t i_trk = 0; i_trk < n_tracks; i_trk++) {

            const auto& track_states_per_track = track_states[i_trk].items;
            ASSERT_EQ(track_states_per_track.size(),
                      plane_positions.size() - 2);

            fit_performance_writer.write(track_states_per_track, det, evt_map);
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
    ::testing::Values(std::make_tuple(1 * detray::unit<scalar>::GeV, 0),
                      std::make_tuple(10 * detray::unit<scalar>::GeV, 0),
                      std::make_tuple(100 * detray::unit<scalar>::GeV, 0)));
