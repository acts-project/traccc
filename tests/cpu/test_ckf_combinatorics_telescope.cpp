/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/finding/finding_algorithm.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"
#include "traccc/simulation/simulator.hpp"
#include "traccc/utils/ranges.hpp"

// Test include(s).
#include "tests/ckf_telescope_test.hpp"
#include "traccc/utils/seed_generator.hpp"

// detray include(s).
#include "detray/io/frontend/detector_reader.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/simulation/event_generator/track_generators.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <filesystem>
#include <string>

using namespace traccc;
// This defines the local frame test suite
TEST_P(CkfCombinatoricsTelescopeTests, Run) {

    // Get the parameters
    const std::string name = std::get<0>(GetParam());
    const std::array<scalar, 3u> origin = std::get<1>(GetParam());
    const std::array<scalar, 3u> origin_stddev = std::get<2>(GetParam());
    const std::array<scalar, 2u> mom_range = std::get<3>(GetParam());
    const std::array<scalar, 2u> eta_range = std::get<4>(GetParam());
    const std::array<scalar, 2u> theta_range = eta_to_theta_range(eta_range);
    const std::array<scalar, 2u> phi_range = std::get<5>(GetParam());
    const scalar charge = std::get<6>(GetParam());
    const unsigned int n_truth_tracks = std::get<7>(GetParam());
    const unsigned int n_events = std::get<8>(GetParam());

    const unsigned int nmax_per_seed = 500;

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
    gen_cfg.charge(charge);
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
        n_events, host_det, field, std::move(generator),
        std::move(smearer_writer_cfg), full_path);
    sim.run();

    /*****************************
     * Do the reconstruction
     *****************************/

    // Seed generator
    seed_generator<host_detector_type> sg(host_det, stddevs);

    // Finding algorithm configuration
    typename traccc::finding_algorithm<rk_stepper_type,
                                       host_navigator_type>::config_type cfg;

    typename traccc::finding_algorithm<
        rk_stepper_type, host_navigator_type>::config_type cfg_limit;
    cfg_limit.max_num_branches_per_initial_seed = nmax_per_seed;

    // Finding algorithm object
    traccc::finding_algorithm<rk_stepper_type, host_navigator_type>
        host_finding(cfg);
    traccc::finding_algorithm<rk_stepper_type, host_navigator_type>
        host_finding_limit(cfg_limit);

    // Iterate over events
    for (std::size_t i_evt = 0; i_evt < n_events; i_evt++) {

        // Truth Track Candidates
        traccc::event_map2 evt_map(i_evt, path, path, path);

        traccc::track_candidate_container_types::host truth_track_candidates =
            evt_map.generate_truth_candidates(sg, host_mr);

        ASSERT_EQ(truth_track_candidates.size(), n_truth_tracks);

        // Prepare truth seeds
        traccc::bound_track_parameters_collection_types::host seeds(&host_mr);
        for (unsigned int i_trk = 0; i_trk < n_truth_tracks; i_trk++) {
            seeds.push_back(truth_track_candidates.at(i_trk).header);
        }
        ASSERT_EQ(seeds.size(), n_truth_tracks);

        // Read measurements
        traccc::io::measurement_reader_output readOut(&host_mr);
        traccc::io::read_measurements(readOut, i_evt, path,
                                      traccc::data_format::csv);
        traccc::measurement_collection_types::host& measurements_per_event =
            readOut.measurements;

        // Run finding
        auto track_candidates =
            host_finding(host_det, field, measurements_per_event, seeds);

        auto track_candidates_limit =
            host_finding_limit(host_det, field, measurements_per_event, seeds);

        // Make sure that the number of found tracks = n_track ^ (n_planes + 1)
        ASSERT_EQ(track_candidates.size(),
                  std::pow(n_truth_tracks, plane_positions.size() + 1));

        ASSERT_EQ(track_candidates_limit.size(),
                  n_truth_tracks * nmax_per_seed);
    }
}

// Testing two identical tracks
INSTANTIATE_TEST_SUITE_P(
    CkfCombinatoricsTelescopeValidation0, CkfCombinatoricsTelescopeTests,
    ::testing::Values(std::make_tuple(
        "telescope_combinatorics_twin", std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 2u>{100.f, 100.f}, std::array<scalar, 2u>{0.f, 0.f},
        std::array<scalar, 2u>{0.f, 0.f}, -1.f, 2, 1)));

// Testing three identical tracks
INSTANTIATE_TEST_SUITE_P(
    CkfCombinatoricsTelescopeValidation1, CkfCombinatoricsTelescopeTests,
    ::testing::Values(std::make_tuple(
        "telescope_combinatorics_trio", std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 2u>{100.f, 100.f}, std::array<scalar, 2u>{0.f, 0.f},
        std::array<scalar, 2u>{0.f, 0.f}, -1.f, 3, 1)));
