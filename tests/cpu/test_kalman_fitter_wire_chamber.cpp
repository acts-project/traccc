/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"
#include "traccc/simulation/measurement_smearer.hpp"
#include "traccc/simulation/simulator.hpp"
#include "traccc/simulation/smearing_writer.hpp"
#include "traccc/utils/ranges.hpp"
#include "traccc/utils/seed_generator.hpp"

// Test include(s).
#include "tests/kalman_fitting_wire_chamber_test.hpp"

// detray include(s).
#include "detray/simulation/event_generator/track_generators.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <filesystem>
#include <string>

using namespace traccc;

TEST_P(KalmanFittingWireChamberTests, Run) {

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
    traccc::fitting_performance_writer::config fit_writer_cfg;
    fit_writer_cfg.file_path = "performance_track_fitting_" + name + ".root";
    traccc::fitting_performance_writer fit_performance_writer(fit_writer_cfg);

    /*****************************
     * Build a drift chamber
     *****************************/
    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;

    // Read back detector file
    const std::string path = name + "/";
    detray::io::detector_reader_config reader_cfg{};
    reader_cfg.add_file(path + "wire_chamber_geometry.json")
        .add_file(path + "wire_chamber_homogeneous_material.json")
        //.add_file("wire_chamber_surface_grids.json")
        .do_check(true);

    const auto [host_det, names] =
        detray::io::read_detector<host_detector_type>(host_mr, reader_cfg);
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
    const std::string full_path = io::data_directory() + path;
    std::filesystem::create_directories(full_path);
    auto sim = traccc::simulator<host_detector_type, b_field_t, generator_type,
                                 writer_type>(
        n_events, host_det, field, std::move(generator),
        std::move(smearer_writer_cfg), full_path);

    // Set constrained step size to 2 mm
    sim.get_config().step_constraint = step_constraint;
    sim.get_config().overstep_tolerance = overstep_tolerance;

    sim.run();

    /***************
     * Run fitting
     ***************/

    // Seed generator
    seed_generator<host_detector_type> sg(host_det, stddevs);

    // Fitting algorithm object
    typename traccc::fitting_algorithm<host_fitter_type>::config_type fit_cfg;
    fit_cfg.step_constraint = step_constraint;
    fit_cfg.overstep_tolerance = overstep_tolerance;
    fit_cfg.mask_tolerance = mask_tolerance;
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
        auto track_states = fitting(host_det, field, track_candidates);

        // Iterator over tracks
        const std::size_t n_tracks = track_states.size();
        ASSERT_EQ(n_tracks, n_truth_tracks);

        for (std::size_t i_trk = 0; i_trk < n_tracks; i_trk++) {

            const auto& fit_info = track_states[i_trk].header;
            const auto& track_states_per_track = track_states[i_trk].items;

            consistency_tests(track_states_per_track);

            ndf_tests(fit_info, track_states_per_track);

            fit_performance_writer.write(track_states_per_track, fit_info,
                                         host_det, evt_map);
        }
    }

    fit_performance_writer.finalize();

    /********************
     * Pull value test
     ********************/

    static const std::vector<std::string> pull_names{
        "pull_d0", "pull_z0", "pull_phi", "pull_theta", "pull_qop"};
    pull_value_tests(fit_writer_cfg.file_path, pull_names);

    /********************
     * Success rate test
     ********************/

    scalar success_rate =
        static_cast<scalar>(n_success) / (n_truth_tracks * n_events);

    ASSERT_GE(success_rate, 0.99f);
    ASSERT_LE(success_rate, 1.00f);

    // Remove the data
    std::filesystem::remove_all(full_path);
}

INSTANTIATE_TEST_SUITE_P(
    KalmanFitWireChamberValidation0, KalmanFittingWireChamberTests,
    ::testing::Values(std::make_tuple(
        "2_GeV", std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 3u>{0.f, 0.f, 0.f}, std::array<scalar, 2u>{2.f, 2.f},
        std::array<scalar, 2u>{-1.f, 1.f},
        std::array<scalar, 2u>{0.f, 2.0f * detray::constant<scalar>::pi}, 100,
        100)));

INSTANTIATE_TEST_SUITE_P(
    KalmanFitWireChamberValidation1, KalmanFittingWireChamberTests,
    ::testing::Values(std::make_tuple(
        "10_GeV", std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 2u>{10.f, 10.f}, std::array<scalar, 2u>{-1.f, 1.f},
        std::array<scalar, 2u>{0.f, 2.0f * detray::constant<scalar>::pi}, 100,
        100)));

INSTANTIATE_TEST_SUITE_P(
    KalmanFitWireChamberValidation2, KalmanFittingWireChamberTests,
    ::testing::Values(std::make_tuple(
        "100_GeV", std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 2u>{100.f, 100.f}, std::array<scalar, 2u>{-1.f, 1.f},
        std::array<scalar, 2u>{0.f, 2.0f * detray::constant<scalar>::pi}, 100,
        100)));
