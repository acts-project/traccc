/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
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
#include "tests/kalman_fitting_telescope_test.hpp"

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

class KalmanFittingHoleCountTests : public KalmanFittingTelescopeTests {};

TEST_P(KalmanFittingHoleCountTests, Run) {

    // Get the parameters
    const std::string name = std::get<0>(GetParam());
    const std::array<scalar, 3u> origin = std::get<1>(GetParam());
    const std::array<scalar, 3u> origin_stddev = std::get<2>(GetParam());
    const std::array<scalar, 2u> mom_range = std::get<3>(GetParam());
    const std::array<scalar, 2u> eta_range = std::get<4>(GetParam());
    const std::array<scalar, 2u> theta_range = eta_to_theta_range(eta_range);
    const std::array<scalar, 2u> phi_range = std::get<5>(GetParam());
    const detray::pdg_particle<scalar> ptc = std::get<6>(GetParam());
    const unsigned int n_truth_tracks = std::get<7>(GetParam());
    const unsigned int n_events = std::get<8>(GetParam());
    const bool random_charge = std::get<9>(GetParam());

    // We only test one track of one event
    ASSERT_EQ(n_truth_tracks, 1u);
    ASSERT_EQ(n_events, 1u);

    /*****************************
     * Build a telescope geometry
     *****************************/

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;

    // Read back detector file
    const std::string path = name + "/";
    detray::io::detector_reader_config reader_cfg{};
    reader_cfg.add_file(path + "telescope_detector_geometry.json")
        .add_file(path + "telescope_detector_homogeneous_material.json");

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
    gen_cfg.randomize_charge(random_charge);
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
    sim.get_config().propagation.navigation.overstep_tolerance =
        -100.f * unit<float>::um;
    sim.get_config().propagation.navigation.max_mask_tolerance =
        1.f * unit<float>::mm;
    sim.run();

    /***************
     * Run fitting
     ***************/

    // Seed generator
    seed_generator<host_detector_type> sg(host_det, stddevs);

    // Fitting algorithm object
    traccc::fitting_config fit_cfg;
    fit_cfg.ptc_hypothesis = ptc;
    fit_cfg.propagation.navigation.overstep_tolerance =
        -100.f * unit<float>::um;
    fit_cfg.propagation.navigation.max_mask_tolerance = 1.f * unit<float>::mm;
    traccc::host::kalman_fitting_algorithm fitting(fit_cfg, host_mr);

    // Event map
    traccc::event_data evt_data(path, 0u, host_mr);

    // Truth Track Candidates
    traccc::track_candidate_container_types::host track_candidates =
        evt_data.generate_truth_candidates(sg, host_mr);
    // Candidate vector
    auto& cands = track_candidates.at(0u).items;

    // Some sanity checks
    ASSERT_EQ(track_candidates.size(), n_truth_tracks);
    const auto n_planes = std::get<11>(GetParam());
    ASSERT_EQ(cands.size(), n_planes);

    // Pop some track candidates to create holes
    // => The number of holes = 8
    ASSERT_TRUE(cands.size() > 8u);
    cands.erase(cands.begin());
    cands.erase(cands.begin());
    cands.erase(cands.begin() + 2);
    cands.erase(cands.begin() + 2);
    cands.erase(cands.begin() + 7);
    cands.pop_back();
    cands.pop_back();
    cands.pop_back();

    // A sanity check on the number of candidiates
    ASSERT_EQ(cands.size(), n_planes - 8u);

    // Run fitting
    auto track_states =
        fitting(host_det, field, traccc::get_data(track_candidates));

    // A sanity check
    const std::size_t n_tracks = track_states.size();
    ASSERT_EQ(n_tracks, n_truth_tracks);

    // Check the number of holes
    // The three holes at the end are not counted as KF aborts once it goes
    // through all track candidates
    const auto& fit_res = track_states.at(0u).header;
    ASSERT_EQ(fit_res.n_holes, 5u);

    // Some sanity checks
    ASSERT_FLOAT_EQ(
        static_cast<float>(fit_res.ndf),
        static_cast<float>(track_states.at(0u).items.size()) * 2.f - 5.f);
}

INSTANTIATE_TEST_SUITE_P(
    KalmanFittingHoleCount, KalmanFittingHoleCountTests,
    ::testing::Values(std::make_tuple(
        "telescope_1_GeV_0_phi_muon", std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 3u>{0.f, 0.f, 0.f}, std::array<scalar, 2u>{1.f, 1.f},
        std::array<scalar, 2u>{0.f, 0.f}, std::array<scalar, 2u>{0.f, 0.f},
        detray::muon<scalar>(), 1, 1, false, 20.f, 20u, 20.f)));
