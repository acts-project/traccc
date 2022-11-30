/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "tests/seed_generator.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"

// detray include(s).
#include "detray/detectors/create_telescope_detector.hpp"
#include "detray/propagator/actors/aborters.hpp"
#include "detray/propagator/actors/parameter_resetter.hpp"
#include "detray/propagator/actors/parameter_transporter.hpp"
#include "detray/propagator/actors/pointwise_material_interactor.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/simulation/simulator.hpp"
#include "detray/simulation/track_generators.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

// ROOT include(s).
#include <TF1.h>

using namespace traccc;

// Kalman Fitting Test
//
// Tuple parameter made of (1) initial particle momentum and (2) initial phi
class KalmanFittingTests
    : public ::testing::TestWithParam<std::tuple<scalar, scalar>> {};

// This defines the local frame test suite
TEST_P(KalmanFittingTests, Run) {

    // Test Parameters
    const scalar p0 = std::get<0>(GetParam());
    const scalar phi0 = std::get<1>(GetParam());

    // Memory resource
    vecmem::host_memory_resource host_mr;

    // Performance writer
    traccc::fitting_performance_writer::config writer_cfg;
    writer_cfg.file_path = "performance_track_fitting_" + std::to_string(p0) +
                           "_GeV_" + std::to_string(phi0) + "_phi" + ".root";

    traccc::fitting_performance_writer fit_performance_writer(writer_cfg);

    /*****************************
     * Build a telescope geometry
     *****************************/

    // Plane alignment direction (aligned to x-axis)
    detray::detail::ray<transform3> traj{{0, 0, 0}, 0, {1, 0, 0}, -1};
    // Position of planes (in mm unit)
    std::vector<scalar> plane_positions = {-10., 20., 40., 60.,  80., 100.,
                                           120., 140, 160, 180., 200.};

    // Detector type
    using detector_type =
        detray::detector<detray::detector_registry::telescope_detector,
                         covfie::field>;

    // B field value and its type
    const vector3 B{2 * detray::unit<scalar>::T, 0, 0};
    using b_field_t = typename detector_type::bfield_type;

    // Create the detector
    const auto mat = detray::silicon_tml<scalar>();
    const scalar thickness = 0.5 * detray::unit<scalar>::mm;

    const detector_type det = create_telescope_detector(
        host_mr,
        b_field_t(b_field_t::backend_t::configuration_t{B[0], B[1], B[2]}),
        plane_positions, traj, 100000. * detray::unit<scalar>::mm,
        100000. * detray::unit<scalar>::mm, mat, thickness);

    /***************************
     * Prepare track candidates
     ***************************/

    // Navigator, stepper, and fitter
    using navigator_type = detray::navigator<decltype(det)>;
    using rk_stepper_type = detray::rk_stepper<b_field_t::view_t, transform3,
                                               detray::constrained_step<>>;

    using fitter_type = kalman_fitter<rk_stepper_type, navigator_type>;

    // Standard deviations for seed track parameters
    std::array<scalar, e_bound_size> stddevs = {
        0.03 * detray::unit<scalar>::mm,
        0.03 * detray::unit<scalar>::mm,
        0.017,
        0.017,
        0.001 / detray::unit<scalar>::GeV,
        1 * detray::unit<scalar>::ns};

    // Seed generator
    seed_generator<rk_stepper_type, navigator_type> sg(det, stddevs);

    // Fitting algorithm object
    fitting_algorithm<fitter_type> fitting;

    // File path
    std::string file_path =
        std::to_string(p0) + "_GeV_" + std::to_string(phi0) + "_phi/";
    std::string full_path =
        "detray_simulation/telescope/kf_validation/" + file_path;

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

    std::array<std::string, 5> pull_names{"pull_d0", "pull_z0", "pull_phi",
                                          "pull_theta", "pull_qop"};

    auto f = fit_performance_writer.get_file();

    TF1 gaus{"gaus", "gaus", -5, 5};
    double fit_par[3];

    for (auto name : pull_names) {

        auto pull_d0 = (TH1F*)f->Get(name.c_str());

        // Set the mean seed to 0
        gaus.SetParameters(1, 0.);
        gaus.SetParLimits(1, -1., 1.);
        // Set the standard deviation seed to 1
        gaus.SetParameters(2, 1.0);
        gaus.SetParLimits(2, 0.5, 2.);

        auto res = pull_d0->Fit("gaus", "Q0S");

        gaus.GetParameters(&fit_par[0]);

        // Mean check
        EXPECT_NEAR(fit_par[1], 0, 0.05) << name << " mean value error";

        // Sigma check
        EXPECT_NEAR(fit_par[2], 1, 0.1) << name << " sigma value error";
    }
}

INSTANTIATE_TEST_SUITE_P(
    KalmanFitValidation, KalmanFittingTests,
    ::testing::Values(std::make_tuple(1 * detray::unit<scalar>::GeV, 0),
                      std::make_tuple(10 * detray::unit<scalar>::GeV, 0),
                      std::make_tuple(100 * detray::unit<scalar>::GeV, 0)));
