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
TEST_P(CkfSparseTrackTelescopeTests, Run) {

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

    // Performance writer
    traccc::fitting_performance_writer::config fit_writer_cfg;
    fit_writer_cfg.file_path = "performance_track_fitting_" + name + ".root";
    traccc::fitting_performance_writer fit_performance_writer(fit_writer_cfg);

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
    sim.run();

    /*****************************
     * Do the reconstruction
     *****************************/

    // Seed generator
    seed_generator<host_detector_type> sg(host_det, stddevs);

    // Finding algorithm configuration
    typename traccc::finding_algorithm<rk_stepper_type,
                                       host_navigator_type>::config_type cfg;
    cfg.ptc_hypothesis = ptc;

    // Finding algorithm object
    traccc::finding_algorithm<rk_stepper_type, host_navigator_type>
        host_finding(cfg);

    // Fitting algorithm object
    typename traccc::fitting_algorithm<host_fitter_type>::config_type fit_cfg;
    fit_cfg.ptc_hypothesis = ptc;
    traccc::fitting_algorithm<host_fitter_type> host_fitting(fit_cfg);

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

        ASSERT_EQ(track_candidates.size(), n_truth_tracks);

        // Run fitting
        auto track_states = host_fitting(host_det, field, track_candidates);

        ASSERT_EQ(track_states.size(), n_truth_tracks);

        for (unsigned int i_trk = 0; i_trk < n_truth_tracks; i_trk++) {

            const auto& track_states_per_track = track_states[i_trk].items;
            const auto& fit_res = track_states[i_trk].header;

            consistency_tests(track_states_per_track);

            ndf_tests(fit_res, track_states_per_track);

            fit_performance_writer.write(track_states_per_track, fit_res,
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

    float success_rate =
        static_cast<float>(n_success) / (n_truth_tracks * n_events);

    ASSERT_FLOAT_EQ(success_rate, 1.00f);
}

INSTANTIATE_TEST_SUITE_P(
    CkfSparseTrackTelescopeValidation0, CkfSparseTrackTelescopeTests,
    ::testing::Values(std::make_tuple("telescope_single_tracks",
                                      std::array<scalar, 3u>{0.f, 0.f, 0.f},
                                      std::array<scalar, 3u>{0.f, 200.f, 200.f},
                                      std::array<scalar, 2u>{1.f, 1.f},
                                      std::array<scalar, 2u>{0.f, 0.f},
                                      std::array<scalar, 2u>{0.f, 0.f},
                                      detray::muon<scalar>(), 1, 5000, false)));

INSTANTIATE_TEST_SUITE_P(
    CkfSparseTrackTelescopeValidation1, CkfSparseTrackTelescopeTests,
    ::testing::Values(std::make_tuple("telescope_double_tracks",
                                      std::array<scalar, 3u>{0.f, 0.f, 0.f},
                                      std::array<scalar, 3u>{0.f, 200.f, 200.f},
                                      std::array<scalar, 2u>{1.f, 1.f},
                                      std::array<scalar, 2u>{0.f, 0.f},
                                      std::array<scalar, 2u>{0.f, 0.f},
                                      detray::muon<scalar>(), 2, 2500, false)));

INSTANTIATE_TEST_SUITE_P(
    CkfSparseTrackTelescopeValidation2, CkfSparseTrackTelescopeTests,
    ::testing::Values(std::make_tuple("telescope_quadra_tracks",
                                      std::array<scalar, 3u>{0.f, 0.f, 0.f},
                                      std::array<scalar, 3u>{0.f, 200.f, 200.f},
                                      std::array<scalar, 2u>{1.f, 1.f},
                                      std::array<scalar, 2u>{0.f, 0.f},
                                      std::array<scalar, 2u>{0.f, 0.f},
                                      detray::muon<scalar>(), 4, 1250, false)));

INSTANTIATE_TEST_SUITE_P(
    CkfSparseTrackTelescopeValidation3, CkfSparseTrackTelescopeTests,
    ::testing::Values(std::make_tuple("telescope_decade_tracks",
                                      std::array<scalar, 3u>{0.f, 0.f, 0.f},
                                      std::array<scalar, 3u>{0.f, 200.f, 200.f},
                                      std::array<scalar, 2u>{1.f, 1.f},
                                      std::array<scalar, 2u>{0.f, 0.f},
                                      std::array<scalar, 2u>{0.f, 0.f},
                                      detray::muon<scalar>(), 10, 500, false)));

INSTANTIATE_TEST_SUITE_P(
    CkfSparseTrackTelescopeValidation4, CkfSparseTrackTelescopeTests,
    ::testing::Values(std::make_tuple("telescope_decade_tracks_random_charge",
                                      std::array<scalar, 3u>{0.f, 0.f, 0.f},
                                      std::array<scalar, 3u>{0.f, 200.f, 200.f},
                                      std::array<scalar, 2u>{1.f, 1.f},
                                      std::array<scalar, 2u>{0.f, 0.f},
                                      std::array<scalar, 2u>{0.f, 0.f},
                                      detray::muon<scalar>(), 10, 500, true)));
