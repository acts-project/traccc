/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/kalman_fitting_algorithm.hpp"
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
#include "detray/io/frontend/detector_reader.hpp"
#include "detray/test/utils/simulation/event_generator/track_generators.hpp"

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
    const detray::pdg_particle<scalar> ptc = std::get<6>(GetParam());
    const unsigned int n_truth_tracks = std::get<7>(GetParam());
    const unsigned int n_events = std::get<8>(GetParam());
    const bool random_charge = std::get<9>(GetParam());

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
        .add_file(path + "wire_chamber_surface_grids.json")
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
    gen_cfg.origin(std::get<1>(GetParam()));
    gen_cfg.origin_stddev(std::get<2>(GetParam()));
    gen_cfg.phi_range(std::get<5>(GetParam()));
    gen_cfg.eta_range(std::get<4>(GetParam()));
    gen_cfg.mom_range(std::get<3>(GetParam()));
    gen_cfg.randomize_charge(random_charge);
    gen_cfg.seed(42);
    generator_type generator(gen_cfg);

    // Smearing value for measurements
    traccc::measurement_smearer<traccc::default_algebra> meas_smearer(
        smearing[0], smearing[1]);

    using writer_type = traccc::smearing_writer<
        traccc::measurement_smearer<traccc::default_algebra>>;

    typename writer_type::config smearer_writer_cfg{meas_smearer};

    // Run simulator
    const std::string full_path = io::data_directory() + path;
    std::filesystem::create_directories(full_path);
    auto sim = traccc::simulator<host_detector_type, b_field_t, generator_type,
                                 writer_type>(
        ptc, n_events, host_det, field, std::move(generator),
        std::move(smearer_writer_cfg), full_path);

    // Set constrained step size to 1 mm
    sim.get_config().propagation.stepping.step_constraint = step_constraint;
    sim.get_config().propagation.navigation.min_mask_tolerance =
        25.f * detray::unit<float>::um;
    sim.get_config().propagation.navigation.search_window = search_window;

    sim.run();

    /***************
     * Run fitting
     ***************/

    // Seed generator
    seed_generator<host_detector_type> sg(host_det, stddevs);

    // Fitting algorithm object
    traccc::fitting_config fit_cfg;
    fit_cfg.propagation.navigation.min_mask_tolerance =
        static_cast<float>(mask_tolerance);
    fit_cfg.propagation.navigation.search_window = search_window;
    fit_cfg.ptc_hypothesis = ptc;
    traccc::host::kalman_fitting_algorithm fitting(fit_cfg, host_mr);

    // Iterate over events
    for (std::size_t i_evt = 0; i_evt < n_events; i_evt++) {

        // Event map
        traccc::event_data evt_data(path, i_evt, host_mr);
        // Truth Track Candidates
        traccc::track_candidate_container_types::host track_candidates =
            evt_data.generate_truth_candidates(sg, host_mr);

        // n_trakcs = 100
        ASSERT_EQ(track_candidates.size(), n_truth_tracks);

        // Run fitting
        auto track_states =
            fitting(host_det, field, traccc::get_data(track_candidates));

        // Iterator over tracks
        const std::size_t n_tracks = track_states.size();
        ASSERT_EQ(n_tracks, n_truth_tracks);

        for (std::size_t i_trk = 0; i_trk < n_tracks; i_trk++) {

            const auto& fit_res = track_states[i_trk].header;
            const auto& track_states_per_track = track_states[i_trk].items;

            consistency_tests(track_states_per_track);

            ndf_tests(fit_res, track_states_per_track);

            fit_performance_writer.write(track_states_per_track, fit_res,
                                         host_det, evt_data);
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

    scalar success_rate = static_cast<scalar>(n_success) /
                          static_cast<scalar>(n_truth_tracks * n_events);

    ASSERT_GE(success_rate, 0.99f);
    ASSERT_LE(success_rate, 1.00f);
}

INSTANTIATE_TEST_SUITE_P(
    KalmanFitWireChamberValidation0, KalmanFittingWireChamberTests,
    ::testing::Values(std::make_tuple(
        "wire_2_GeV_muon", std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 3u>{0.f, 0.f, 0.f}, std::array<scalar, 2u>{2.f, 2.f},
        std::array<scalar, 2u>{-1.f, 1.f},
        std::array<scalar, 2u>{-detray::constant<scalar>::pi,
                               detray::constant<scalar>::pi},
        detray::muon<scalar>(), 100, 100, false)));

// @TODO: Make full eta range work
INSTANTIATE_TEST_SUITE_P(
    KalmanFitWireChamberValidation1, KalmanFittingWireChamberTests,
    ::testing::Values(std::make_tuple(
        "wire_10_GeV_muon", std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 2u>{10.f, 10.f}, std::array<scalar, 2u>{-0.3f, 0.3f},
        std::array<scalar, 2u>{-detray::constant<scalar>::pi,
                               detray::constant<scalar>::pi},
        detray::muon<scalar>(), 100, 100, false)));

// @TODO: Make full eta range work
INSTANTIATE_TEST_SUITE_P(
    KalmanFitWireChamberValidation2, KalmanFittingWireChamberTests,
    ::testing::Values(std::make_tuple(
        "wire_100_GeV_muon", std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 2u>{100.f, 100.f},
        std::array<scalar, 2u>{-0.4f, 0.4f},
        std::array<scalar, 2u>{-detray::constant<scalar>::pi,
                               detray::constant<scalar>::pi},
        detray::muon<scalar>(), 100, 100, false)));

INSTANTIATE_TEST_SUITE_P(
    KalmanFitWireChamberValidation3, KalmanFittingWireChamberTests,
    ::testing::Values(std::make_tuple(
        "wire_2_GeV_anti_muon", std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 3u>{0.f, 0.f, 0.f}, std::array<scalar, 2u>{2.f, 2.f},
        std::array<scalar, 2u>{-1.f, 1.f},
        std::array<scalar, 2u>{-detray::constant<scalar>::pi,
                               detray::constant<scalar>::pi},
        detray::antimuon<scalar>(), 100, 100, false)));

INSTANTIATE_TEST_SUITE_P(
    KalmanFitWireChamberValidation4, KalmanFittingWireChamberTests,
    ::testing::Values(std::make_tuple(
        "wire_2_GeV_random_charge", std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 3u>{0.f, 0.f, 0.f}, std::array<scalar, 2u>{2.f, 2.f},
        std::array<scalar, 2u>{-1.f, 1.f},
        std::array<scalar, 2u>{-detray::constant<scalar>::pi,
                               detray::constant<scalar>::pi},
        detray::antimuon<scalar>(), 100, 100, true)));
