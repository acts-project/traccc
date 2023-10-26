/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"
#include "traccc/simulation/simulator.hpp"
#include "traccc/utils/ranges.hpp"
#include "traccc/utils/seed_generator.hpp"

// Test include(s).
#include "tests/kalman_fitting_test.hpp"

// detray include(s).
#include "detray/detectors/create_telescope_detector.hpp"
#include "detray/simulation/event_generator/track_generators.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <filesystem>
#include <string>

using namespace traccc;

TEST_P(KalmanFittingTests, Run) {

    // Get the parameters
    const std::string name = std::get<0>(GetParam());
    const std::array<scalar, 3u> origin = std::get<1>(GetParam());
    const std::array<scalar, 3u> origin_stddev = std::get<2>(GetParam());
    const std::array<scalar, 2u> mom_range = std::get<3>(GetParam());
    const std::array<scalar, 2u> eta_range = std::get<4>(GetParam());
    const std::array<scalar, 2u> theta_range = eta_to_theta_range(eta_range);
    const std::array<scalar, 2u> phi_range = std::get<5>(GetParam());
    const unsigned int n_truth_tracks = std::get<6>(GetParam());
    const unsigned int n_events = std::get<7>(GetParam());

    // Performance writer
    traccc::fitting_performance_writer::config writer_cfg;
    writer_cfg.file_path = "performance_track_fitting_" + name + ".root";
    traccc::fitting_performance_writer fit_performance_writer(writer_cfg);

    /*****************************
     * Build a telescope geometry
     *****************************/

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;

    detray::tel_det_config<> tel_cfg{rectangle};
    tel_cfg.positions(plane_positions);
    tel_cfg.module_material(mat);
    tel_cfg.mat_thickness(thickness);
    tel_cfg.pilot_track(traj);

    auto [det, name_map] = create_telescope_detector(host_mr, tel_cfg);
    auto field = detray::bfield::create_const_field(B);

    /***************************
     * Generate simulation data
     ***************************/

    // Track generator
    using generator_type =
        detray::random_track_generator<traccc::free_track_parameters,
                                       uniform_gen_t>;
    generator_type::configuration gen_cfg{};
    gen_cfg.n_tracks(n_truth_tracks);
    gen_cfg.origin(origin);
    gen_cfg.origin_stddev(origin_stddev);
    gen_cfg.phi_range(phi_range[0], phi_range[1]);
    gen_cfg.theta_range(theta_range[0], theta_range[1]);
    gen_cfg.mom_range(mom_range[0], mom_range[1]);
    generator_type generator(gen_cfg);

    // Smearing value for measurements
    traccc::measurement_smearer<transform3> meas_smearer(smearing[0],
                                                         smearing[1]);

    using writer_type =
        traccc::smearing_writer<traccc::measurement_smearer<transform3>>;

    typename writer_type::config smearer_writer_cfg{meas_smearer};

    // Run simulator
    const std::string path = name + "/";
    const std::string full_path = io::data_directory() + path;
    std::filesystem::create_directories(full_path);
    auto sim = traccc::simulator<host_detector_type, b_field_t, generator_type,
                                 writer_type>(
        n_events, det, field, std::move(generator),
        std::move(smearer_writer_cfg), full_path);
    sim.run();

    /***************
     * Run fitting
     ***************/

    // Seed generator
    seed_generator<host_detector_type> sg(det, stddevs);

    // Fitting algorithm object
    typename traccc::fitting_algorithm<host_fitter_type>::config_type fit_cfg;
    // fit_cfg.n_iterations = 10;
    fitting_algorithm<host_fitter_type> fitting(fit_cfg);

    // Iterate over events
    for (std::size_t i_evt = 0; i_evt < n_events; i_evt++) {
        // Event map
        traccc::event_map2 evt_map(i_evt, path, path, path);

        // Truth Track Candidates
        traccc::track_candidate_container_types::host track_candidates =
            evt_map.generate_truth_candidates(sg, host_mr);

        // n_trakcs = 100
        ASSERT_EQ(track_candidates.size(), n_truth_tracks);

        // Run fitting
        auto track_states = fitting(det, field, track_candidates);

        // Iterator over tracks
        const std::size_t n_tracks = track_states.size();

        // n_trakcs = 100
        ASSERT_EQ(n_tracks, n_truth_tracks);
        for (std::size_t i_trk = 0; i_trk < n_tracks; i_trk++) {

            const auto& track_states_per_track = track_states[i_trk].items;
            ASSERT_EQ(track_states_per_track.size(), plane_positions.size());
            const auto& fit_info = track_states[i_trk].header;
            ASSERT_FLOAT_EQ(fit_info.ndf, 2 * plane_positions.size() - 5.f);

            fit_performance_writer.write(track_states_per_track, fit_info, det,
                                         evt_map);
        }
    }

    fit_performance_writer.finalize();

    /********************
     * Pull value test
     ********************/

    static const std::vector<std::string> pull_names{
        "pull_d0", "pull_z0", "pull_phi", "pull_theta", "pull_qop"};
    pull_value_tests(writer_cfg.file_path, pull_names);

    // Remove the data
    std::filesystem::remove_all(full_path);
}

INSTANTIATE_TEST_SUITE_P(
    KalmanFitValidation0, KalmanFittingTests,
    ::testing::Values(std::make_tuple(
        "1_GeV_0_phi", std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 3u>{0.f, 0.f, 0.f}, std::array<scalar, 2u>{1.f, 1.f},
        std::array<scalar, 2u>{0.f, 0.f}, std::array<scalar, 2u>{0.f, 0.f}, 100,
        100)));

INSTANTIATE_TEST_SUITE_P(
    KalmanFitValidation1, KalmanFittingTests,
    ::testing::Values(std::make_tuple(
        "10_GeV_0_phi", std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 2u>{10.f, 10.f}, std::array<scalar, 2u>{0.f, 0.f},
        std::array<scalar, 2u>{0.f, 0.f}, 100, 100)));

INSTANTIATE_TEST_SUITE_P(
    KalmanFitValidation2, KalmanFittingTests,
    ::testing::Values(std::make_tuple(
        "100_GeV_0_phi", std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 2u>{100.f, 100.f}, std::array<scalar, 2u>{0.f, 0.f},
        std::array<scalar, 2u>{0.f, 0.f}, 100, 100)));