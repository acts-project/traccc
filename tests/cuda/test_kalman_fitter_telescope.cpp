/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/bfield/construct_const_bfield.hpp"
#include "traccc/bfield/magnetic_field_types.hpp"
#include "traccc/cuda/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"
#include "traccc/device/container_h2d_copy_alg.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/performance/details/is_same_object.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"
#include "traccc/simulation/event_generators.hpp"
#include "traccc/simulation/simulator.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/ranges.hpp"
#include "traccc/utils/seed_generator.hpp"

// Test include(s).
#include "tests/kalman_fitting_telescope_test.hpp"

// detray include(s).
#include <detray/io/frontend/detector_reader.hpp>

// VecMem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <filesystem>
#include <string>

using namespace traccc;

// This defines the local frame test suite
TEST_P(KalmanFittingTelescopeTests, Run) {

    // Get the parameters
    const std::string name = std::get<0>(GetParam());
    const std::array<scalar, 3u> origin = std::get<1>(GetParam());
    const std::array<scalar, 3u> origin_stddev = std::get<2>(GetParam());
    const std::array<scalar, 2u> mom_range = std::get<3>(GetParam());
    const std::array<scalar, 2u> eta_range = std::get<4>(GetParam());
    const std::array<scalar, 2u> theta_range = eta_to_theta_range(eta_range);
    const std::array<scalar, 2u> phi_range = std::get<5>(GetParam());
    const traccc::pdg_particle<scalar> ptc = std::get<6>(GetParam());
    const unsigned int n_truth_tracks = std::get<7>(GetParam());
    const unsigned int n_events = std::get<8>(GetParam());
    const bool random_charge = std::get<9>(GetParam());

    // Performance writer
    traccc::fitting_performance_writer::config fit_writer_cfg;
    fit_writer_cfg.file_path = "performance_track_fitting_" + name + ".root";
    traccc::fitting_performance_writer fit_performance_writer(
        fit_writer_cfg, traccc::getDefaultLogger("FittingPerformanceWriter",
                                                 traccc::Logging::Level::INFO));

    /*****************************
     * Build a telescope geometry
     *****************************/

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &host_mr};
    vecmem::cuda::managed_memory_resource mng_mr;

    // Read back detector file
    const std::string path = name + "/";
    detray::io::detector_reader_config reader_cfg{};
    reader_cfg.add_file(path + "telescope_detector_geometry.json")
        .add_file(path + "telescope_detector_homogeneous_material.json");
    auto [host_det, names] =
        detray::io::read_detector<host_detector_type>(mng_mr, reader_cfg);

    // Detector view object
    auto det_view = detray::get_data(host_det);

    const auto field = traccc::construct_const_bfield(std::get<13>(GetParam()));

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
        ptc, n_events, host_det,
        field.as_field<traccc::const_bfield_backend_t<traccc::scalar>>(),
        std::move(generator), std::move(smearer_writer_cfg), full_path);
    sim.run();

    /***************
     * Run fitting
     ***************/

    // Stream object
    traccc::cuda::stream stream;

    // Copy objects
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    traccc::device::container_d2h_copy_alg<traccc::track_state_container_types>
        track_state_d2h{mr, copy};

    // Seed generator
    seed_generator<host_detector_type> sg(host_det, stddevs);

    // Fitting algorithm object
    traccc::cuda::kalman_fitting_algorithm::config_type fit_cfg;
    fit_cfg.ptc_hypothesis = ptc;
    traccc::cuda::kalman_fitting_algorithm device_fitting(fit_cfg, mr, copy,
                                                          stream);

    // Iterate over events
    for (std::size_t i_evt = 0; i_evt < n_events; i_evt++) {
        // Event map
        traccc::event_data evt_data(path, i_evt, host_mr);

        // Truth Track Candidates
        traccc::edm::track_candidate_container<traccc::default_algebra>::host
            track_candidates{host_mr};
        evt_data.generate_truth_candidates(track_candidates, sg, host_mr);

        // n_trakcs = 100
        ASSERT_EQ(track_candidates.tracks.size(), n_truth_tracks);

        // track candidates buffer
        traccc::edm::track_candidate_container<traccc::default_algebra>::buffer
            track_candidates_buffer{
                copy.to(vecmem::get_data(track_candidates.tracks), mr.main,
                        mr.host, vecmem::copy::type::host_to_device),
                copy.to(vecmem::get_data(track_candidates.measurements),
                        mr.main, vecmem::copy::type::host_to_device)};

        // Run fitting
        traccc::track_state_container_types::buffer track_states_cuda_buffer =
            device_fitting(det_view, field,
                           {track_candidates_buffer.tracks,
                            track_candidates_buffer.measurements});

        traccc::track_state_container_types::host track_states_cuda =
            track_state_d2h(track_states_cuda_buffer);
        const std::size_t n_fitted_tracks =
            count_successfully_fitted_tracks(track_states_cuda);

        ASSERT_EQ(track_states_cuda.size(), n_truth_tracks);
        ASSERT_EQ(track_states_cuda.size(), n_fitted_tracks);

        for (std::size_t i_trk = 0; i_trk < n_truth_tracks; i_trk++) {

            const auto& track_states_per_track = track_states_cuda[i_trk].items;
            const auto& fit_res = track_states_cuda[i_trk].header;

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
     * P-value test
     ********************/

    p_value_tests(fit_writer_cfg.file_path);

    /********************
     * Success rate test
     ********************/

    float success_rate = static_cast<float>(n_success) /
                         static_cast<float>(n_truth_tracks * n_events);

    ASSERT_FLOAT_EQ(success_rate, 1.00f);
}

INSTANTIATE_TEST_SUITE_P(
    CUDAKalmanFitTelescopeValidation, KalmanFittingTelescopeTests,
    ::testing::Values(
        std::make_tuple(
            "cuda_telescope_1_GeV_0_phi", std::array<scalar, 3u>{0.f, 0.f, 0.f},
            std::array<scalar, 3u>{0.f, 0.f, 0.f},
            std::array<scalar, 2u>{1.f, 1.f}, std::array<scalar, 2u>{0.f, 0.f},
            std::array<scalar, 2u>{0.f, 0.f}, traccc::muon<scalar>(), 100, 100,
            false, 20.f, 9u, 20.f, vector3{0, 0, 2 * traccc::unit<scalar>::T}),
        std::make_tuple("cuda_telescope_10_GeV_0_phi",
                        std::array<scalar, 3u>{0.f, 0.f, 0.f},
                        std::array<scalar, 3u>{0.f, 0.f, 0.f},
                        std::array<scalar, 2u>{10.f, 10.f},
                        std::array<scalar, 2u>{0.f, 0.f},
                        std::array<scalar, 2u>{0.f, 0.f},
                        traccc::muon<scalar>(), 100, 100, false, 20.f, 9u, 20.f,
                        vector3{0, 0, 2 * traccc::unit<scalar>::T}),
        std::make_tuple("cuda_telescope_100_GeV_0_phi",
                        std::array<scalar, 3u>{0.f, 0.f, 0.f},
                        std::array<scalar, 3u>{0.f, 0.f, 0.f},
                        std::array<scalar, 2u>{100.f, 100.f},
                        std::array<scalar, 2u>{0.f, 0.f},
                        std::array<scalar, 2u>{0.f, 0.f},
                        traccc::muon<scalar>(), 100, 100, false, 20.f, 9u, 20.f,
                        vector3{0, 0, 2 * traccc::unit<scalar>::T}),
        std::make_tuple("cuda_telescope_1_GeV_0_phi_antimuon",
                        std::array<scalar, 3u>{0.f, 0.f, 0.f},
                        std::array<scalar, 3u>{0.f, 0.f, 0.f},
                        std::array<scalar, 2u>{1.f, 1.f},
                        std::array<scalar, 2u>{0.f, 0.f},
                        std::array<scalar, 2u>{0.f, 0.f},
                        traccc::antimuon<scalar>(), 100, 100, false, 20.f, 9u,
                        20.f, vector3{0, 0, 2 * traccc::unit<scalar>::T}),
        std::make_tuple("cuda_telescope_1_GeV_0_phi_random_charge",
                        std::array<scalar, 3u>{0.f, 0.f, 0.f},
                        std::array<scalar, 3u>{0.f, 0.f, 0.f},
                        std::array<scalar, 2u>{1.f, 1.f},
                        std::array<scalar, 2u>{0.f, 0.f},
                        std::array<scalar, 2u>{0.f, 0.f},
                        traccc::muon<scalar>(), 100, 100, true, 20.f, 9u, 20.f,
                        vector3{0, 0, 2 * traccc::unit<scalar>::T})));
