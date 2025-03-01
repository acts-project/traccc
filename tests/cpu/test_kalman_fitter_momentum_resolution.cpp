/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"
#include "traccc/simulation/simulator.hpp"
#include "traccc/utils/ranges.hpp"
#include "traccc/utils/seed_generator.hpp"

// Test include(s).
#include "tests/kalman_fitting_momentum_resolution_test.hpp"

// detray include(s).
#include <detray/io/frontend/detector_reader.hpp>
#include <detray/test/utils/simulation/event_generator/track_generators.hpp>

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <filesystem>
#include <string>

using namespace traccc;

TEST_P(KalmanFittingMomentumResolutionTests, Run) {

    // Get the parameters
    const std::string name = std::get<0>(GetParam());
    const std::array<scalar, 3u> origin = std::get<1>(GetParam());
    const std::array<scalar, 3u> origin_stddev = std::get<2>(GetParam());
    const scalar p = std::get<3>(GetParam());
    const scalar eta = std::get<4>(GetParam());
    const scalar theta = eta_to_theta(eta);
    const scalar phi = std::get<5>(GetParam());
    const detray::pdg_particle<scalar> ptc = std::get<6>(GetParam());
    const unsigned int n_truth_tracks = std::get<7>(GetParam());
    const unsigned int n_events = std::get<8>(GetParam());
    const bool random_charge = std::get<9>(GetParam());

    // Performance writer
    traccc::fitting_performance_writer::config fit_writer_cfg;
    fit_writer_cfg.res_config.var_binning["residual_qopT"] =
        plot_helpers::binning("r_{q/p_{T}} [c/GeV]", 1000, -0.1f, 0.1f);
    fit_writer_cfg.file_path = "performance_track_fitting_" + name + ".root";
    traccc::fitting_performance_writer fit_performance_writer(fit_writer_cfg);

    // Set qop stddev to 10% of truth qop
    const scalar qop_stddev = 0.1f / p;

    stddevs[4] = qop_stddev;

    /*****************************
     * Build a telescope geometry
     *****************************/

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;

    // Read back detector file
    const std::string path = name + "/";
    detray::io::detector_reader_config reader_cfg{};
    reader_cfg.add_file(path + "telescope_detector_geometry.json");
    // If the module material is not vacuum, read the material map
    if (std::get<14>(GetParam()) != detray::vacuum<scalar>()) {
        reader_cfg.add_file(path +
                            "telescope_detector_homogeneous_material.json");
    }

    const auto [host_det, names] =
        detray::io::read_detector<host_detector_type>(host_mr, reader_cfg);
    auto field =
        detray::bfield::create_const_field<scalar>(std::get<13>(GetParam()));

    const auto vol0 = detray::tracking_volume{host_det, 0u};

    // The number of sensitive surfaces = # of total surfaces - # of portals
    // (=6)
    const std::size_t n_sensitive_surfaces = vol0.surfaces().size() - 6u;

    ASSERT_EQ(n_sensitive_surfaces, std::get<11>(GetParam()));

    /***************************
     * Generate simulation data
     ***************************/

    // Track generator
    using generator_type =
        detray::random_track_generator<traccc::free_track_parameters<>,
                                       uniform_gen_t>;
    generator_type::configuration gen_cfg{};
    gen_cfg.n_tracks(n_truth_tracks);
    gen_cfg.origin(origin);
    gen_cfg.origin_stddev(origin_stddev);
    gen_cfg.phi_range(phi, phi);
    gen_cfg.theta_range(theta, theta);
    gen_cfg.mom_range(p, p);
    gen_cfg.randomize_charge(random_charge);
    generator_type generator(gen_cfg);

    // Smearing value for measurements
    const auto smearing{std::get<15>(GetParam())};
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
    sim.run();

    /***************
     * Run fitting
     ***************/

    // Seed generator
    seed_generator<host_detector_type> sg(host_det, stddevs);

    // Fitting algorithm object
    traccc::fitting_config fit_cfg;
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

        // The nubmer of track candidates per track should be equal to the
        // number of planes
        for (std::size_t i_trk = 0; i_trk < n_truth_tracks; i_trk++) {
            ASSERT_EQ(track_candidates.at(i_trk).items.size(),
                      std::get<11>(GetParam()));
        }

        // Run fitting
        auto track_states =
            fitting(host_det, field, traccc::get_data(track_candidates));

        // Iterator over tracks
        const std::size_t n_tracks = track_states.size();
        const std::size_t n_fitted_tracks = count_fitted_tracks(track_states);

        // n_trakcs = 100
        ASSERT_GE(static_cast<float>(n_tracks),
                  0.98 * static_cast<float>(n_truth_tracks));
        ASSERT_GE(static_cast<float>(n_tracks),
                  0.98 * static_cast<float>(n_fitted_tracks));

        for (std::size_t i_trk = 0; i_trk < n_tracks; i_trk++) {

            const auto& track_states_per_track = track_states[i_trk].items;
            const auto& fit_res = track_states[i_trk].header;

            consistency_tests(track_states_per_track);

            ndf_tests(fit_res, track_states_per_track);

            ASSERT_EQ(fit_res.trk_quality.n_holes, 0u);

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
     * P-value test
     ********************/

    p_value_tests(fit_writer_cfg.file_path);

    //**************************/
    // Momentum resolution test
    //**************************/

    momentum_resolution_tests(fit_writer_cfg.file_path);

    /********************
     * Success rate test
     ********************/

    float success_rate = static_cast<float>(n_success) /
                         static_cast<float>(n_truth_tracks * n_events);

    ASSERT_GE(success_rate, 0.98f);
}

// Muon with 1, 10, 100 GeV/c, no materials
INSTANTIATE_TEST_SUITE_P(
    KalmanFitMomentumResolutionValidation0,
    KalmanFittingMomentumResolutionTests,
    ::testing::Values(
        std::make_tuple(
            "mom_resolution_1_GeV_muon", std::array<scalar, 3u>{0.f, 0.f, 0.f},
            std::array<scalar, 3u>{0.f, 0.f, 0.f}, 1.f, 0.f, 0.f,
            detray::muon<scalar>(), 100, 100, false, 20.f, 20u, 50.f,
            vector3{0, 0, 2 * traccc::unit<scalar>::T},
            detray::vacuum<scalar>(),
            std::array<scalar, 2u>{50.f * traccc::unit<scalar>::um,
                                   50.f * traccc::unit<scalar>::um}),
        std::make_tuple(
            "mom_resolution_10_GeV_muon", std::array<scalar, 3u>{0.f, 0.f, 0.f},
            std::array<scalar, 3u>{0.f, 0.f, 0.f}, 10.f, 0.f, 0.f,
            detray::muon<scalar>(), 100, 100, false, 20.f, 20u, 50.f,
            vector3{0, 0, 2 * traccc::unit<scalar>::T},
            detray::vacuum<scalar>(),
            std::array<scalar, 2u>{50.f * traccc::unit<scalar>::um,
                                   50.f * traccc::unit<scalar>::um}),
        std::make_tuple("mom_resolution_100_GeV_muon",
                        std::array<scalar, 3u>{0.f, 0.f, 0.f},
                        std::array<scalar, 3u>{0.f, 0.f, 0.f}, 100.f, 0.f, 0.f,
                        detray::muon<scalar>(), 100, 100, false, 20.f, 20u,
                        50.f, vector3{0, 0, 2 * traccc::unit<scalar>::T},
                        detray::vacuum<scalar>(),
                        std::array<scalar, 2u>{
                            50.f * traccc::unit<scalar>::um,
                            50.f * traccc::unit<scalar>::um})));

// Muon with 1 GeV/c and different smearing parameters, no materials
INSTANTIATE_TEST_SUITE_P(
    KalmanFitMomentumResolutionValidation1,
    KalmanFittingMomentumResolutionTests,
    ::testing::Values(
        std::make_tuple("mom_resolution_1_GeV_muon_50_100_smearing",
                        std::array<scalar, 3u>{0.f, 0.f, 0.f},
                        std::array<scalar, 3u>{0.f, 0.f, 0.f}, 1.f, 0.f, 0.f,
                        detray::muon<scalar>(), 100, 100, false, 20.f, 20u,
                        50.f, vector3{0, 0, 2 * traccc::unit<scalar>::T},
                        detray::vacuum<scalar>(),
                        std::array<scalar, 2u>{
                            50.f * traccc::unit<scalar>::um,
                            100.f * traccc::unit<scalar>::um}),
        std::make_tuple("mom_resolution_1_GeV_muon_100_50_smearing",
                        std::array<scalar, 3u>{0.f, 0.f, 0.f},
                        std::array<scalar, 3u>{0.f, 0.f, 0.f}, 1.f, 0.f, 0.f,
                        detray::muon<scalar>(), 100, 100, false, 20.f, 20u,
                        50.f, vector3{0, 0, 2 * traccc::unit<scalar>::T},
                        detray::vacuum<scalar>(),
                        std::array<scalar, 2u>{
                            100.f * traccc::unit<scalar>::um,
                            50.f * traccc::unit<scalar>::um}),
        std::make_tuple("mom_resolution_1_GeV_muon_100_100_smearing",
                        std::array<scalar, 3u>{0.f, 0.f, 0.f},
                        std::array<scalar, 3u>{0.f, 0.f, 0.f}, 1.f, 0.f, 0.f,
                        detray::muon<scalar>(), 100, 100, false, 20.f, 20u,
                        50.f, vector3{0, 0, 2 * traccc::unit<scalar>::T},
                        detray::vacuum<scalar>(),
                        std::array<scalar, 2u>{
                            100.f * traccc::unit<scalar>::um,
                            100.f * traccc::unit<scalar>::um})));
