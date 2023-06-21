/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/cuda/finding/finding_algorithm.hpp"
#include "traccc/cuda/fitting/fitting_algorithm.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"
#include "traccc/device/container_h2d_copy_alg.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/ranges.hpp"
#include "traccc/utils/seed_generator.hpp"

// Test include(s).
#include "tests/combinatorial_kalman_finding_test.hpp"

// detray include(s).
#include "detray/detectors/create_telescope_detector.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/simulation/event_generator/track_generators.hpp"
#include "detray/simulation/simulator.hpp"

// VecMem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <filesystem>
#include <string>

using namespace traccc;
// This defines the local frame test suite
TEST_P(CkfSparseTrackTests, Run) {

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
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &host_mr};
    vecmem::cuda::managed_memory_resource mng_mr;

    host_detector_type host_det = create_telescope_detector(
        mng_mr,
        b_field_t(b_field_t::backend_t::configuration_t{B[0], B[1], B[2]}),
        rectangle, plane_positions, mat, thickness, traj);

    // Detector view object
    auto det_view = detray::get_data(host_det);

    /***************************
     * Generate simulation data
     ***************************/

    auto generator =
        detray::random_track_generator<traccc::free_track_parameters,
                                       uniform_gen_t>(n_truth_tracks, origin,
                                                      origin_stddev, mom_range,
                                                      theta_range, phi_range);

    // Smearing value for measurements
    detray::measurement_smearer<transform3> meas_smearer(smearing[0],
                                                         smearing[1]);

    // Run simulator
    const std::string path = name + "/";
    const std::string full_path = io::data_directory() + path;
    std::filesystem::create_directories(full_path);
    auto sim = detray::simulator(n_events, host_det, std::move(generator),
                                 meas_smearer, full_path);
    sim.run();

    /*****************************
     * Do the reconstruction
     *****************************/

    // Copy objects
    vecmem::cuda::copy copy;

    traccc::device::container_h2d_copy_alg<traccc::measurement_container_types>
        measurement_h2d{mr, copy};

    traccc::device::container_d2h_copy_alg<
        traccc::track_candidate_container_types>
        track_candidate_d2h{mr, copy};

    traccc::device::container_d2h_copy_alg<traccc::track_state_container_types>
        track_state_d2h{mr, copy};

    // Seed generator
    seed_generator<host_detector_type> sg(host_det, stddevs);

    // Finding algorithm configuration
    typename traccc::cuda::finding_algorithm<
        rk_stepper_type, device_navigator_type>::config_type cfg;
    // few tracks (~1 out of 1000 tracks) are missed when chi2_max = 15
    cfg.chi2_max = 30.f;

    // Finding algorithm object
    traccc::cuda::finding_algorithm<rk_stepper_type, device_navigator_type>
        device_finding(cfg, mr);

    // Fitting algorithm object
    typename traccc::cuda::fitting_algorithm<device_fitter_type>::config_type
        fit_cfg;
    traccc::cuda::fitting_algorithm<device_fitter_type> device_fitting(fit_cfg,
                                                                       mr);

    // Iterate over events
    for (std::size_t i_evt = 0; i_evt < n_events; i_evt++) {

        // Truth Track Candidates
        traccc::event_map2 evt_map(i_evt, path, path, path);

        traccc::track_candidate_container_types::host truth_track_candidates =
            evt_map.generate_truth_candidates(sg, mng_mr);

        ASSERT_EQ(truth_track_candidates.size(), n_truth_tracks);

        // Prepare truth seeds
        traccc::bound_track_parameters_collection_types::host seeds(mr.host);
        for (unsigned int i_trk = 0; i_trk < n_truth_tracks; i_trk++) {
            seeds.push_back(truth_track_candidates.at(i_trk).header);
        }
        ASSERT_EQ(seeds.size(), n_truth_tracks);

        traccc::bound_track_parameters_collection_types::buffer seeds_buffer{
            static_cast<unsigned int>(seeds.size()), mr.main};
        copy.setup(seeds_buffer);
        copy(vecmem::get_data(seeds), seeds_buffer,
             vecmem::copy::type::host_to_device);

        // Read measurements
        traccc::measurement_container_types::host measurements_per_event =
            traccc::io::read_measurements_container(
                i_evt, path, traccc::data_format::csv, &host_mr);
        traccc::measurement_container_types::buffer measurements_buffer =
            measurement_h2d(traccc::get_data(measurements_per_event));

        // Instantiate output cuda containers/collections
        traccc::track_candidate_container_types::buffer
            track_candidates_cuda_buffer{{{}, *(mr.host)},
                                         {{}, *(mr.host), mr.host}};
        copy.setup(track_candidates_cuda_buffer.headers);
        copy.setup(track_candidates_cuda_buffer.items);

        // Navigation buffer
        auto navigation_buffer = detray::create_candidates_buffer(
            host_det,
            device_finding.get_config().max_num_branches_per_seed *
                seeds.size(),
            mr.main, mr.host);

        // Run finding
        track_candidates_cuda_buffer =
            device_finding(det_view, navigation_buffer, measurements_buffer,
                           std::move(seeds_buffer));

        traccc::track_candidate_container_types::host track_candidates_cuda =
            track_candidate_d2h(track_candidates_cuda_buffer);

        ASSERT_EQ(track_candidates_cuda.size(), n_truth_tracks);

        // Instantiate cuda containers/collections
        traccc::track_state_container_types::buffer track_states_cuda_buffer{
            {{}, *(mr.host)}, {{}, *(mr.host), mr.host}};

        // Run fitting
        track_states_cuda_buffer = device_fitting(det_view, navigation_buffer,
                                                  track_candidates_cuda_buffer);

        traccc::track_state_container_types::host track_states_cuda =
            track_state_d2h(track_states_cuda_buffer);

        ASSERT_EQ(track_states_cuda.size(), n_truth_tracks);

        for (unsigned int i = 0; i < n_truth_tracks; i++) {
            const auto& trk_states_per_track = track_states_cuda.at(i).items;
            ASSERT_EQ(trk_states_per_track.size(), plane_positions.size());
            const auto& fit_info = track_states_cuda[i].header;
            ASSERT_FLOAT_EQ(fit_info.ndf, 2 * plane_positions.size() - 5.f);

            fit_performance_writer.write(trk_states_per_track, fit_info,
                                         host_det, evt_map);
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
    CkfSparseTrackValidation0, CkfSparseTrackTests,
    ::testing::Values(std::make_tuple(
        "single_tracks", std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 3u>{0.f, 200.f, 200.f},
        std::array<scalar, 2u>{1.f, 1.f}, std::array<scalar, 2u>{0.f, 0.f},
        std::array<scalar, 2u>{0.f, 0.f}, 1, 5000)));

INSTANTIATE_TEST_SUITE_P(
    CkfSparseTrackValidation1, CkfSparseTrackTests,
    ::testing::Values(std::make_tuple(
        "double_tracks", std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 3u>{0.f, 200.f, 200.f},
        std::array<scalar, 2u>{1.f, 1.f}, std::array<scalar, 2u>{0.f, 0.f},
        std::array<scalar, 2u>{0.f, 0.f}, 2, 2500)));

INSTANTIATE_TEST_SUITE_P(
    CkfSparseTrackValidation2, CkfSparseTrackTests,
    ::testing::Values(std::make_tuple(
        "quadra_tracks", std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 3u>{0.f, 200.f, 200.f},
        std::array<scalar, 2u>{1.f, 1.f}, std::array<scalar, 2u>{0.f, 0.f},
        std::array<scalar, 2u>{0.f, 0.f}, 4, 1250)));

INSTANTIATE_TEST_SUITE_P(
    CkfSparseTrackValidation3, CkfSparseTrackTests,
    ::testing::Values(std::make_tuple(
        "decade_tracks", std::array<scalar, 3u>{0.f, 0.f, 0.f},
        std::array<scalar, 3u>{0.f, 200.f, 200.f},
        std::array<scalar, 2u>{1.f, 1.f}, std::array<scalar, 2u>{0.f, 0.f},
        std::array<scalar, 2u>{0.f, 0.f}, 10, 500)));