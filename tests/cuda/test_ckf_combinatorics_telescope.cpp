/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/cuda/finding/finding_algorithm.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"
#include "traccc/device/container_h2d_copy_alg.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/simulation/simulator.hpp"
#include "traccc/utils/event_data.hpp"
#include "traccc/utils/ranges.hpp"

// Test include(s).
#include "tests/ckf_telescope_test.hpp"
#include "traccc/utils/seed_generator.hpp"

// detray include(s).
#include <detray/io/frontend/detector_reader.hpp>
#include <detray/propagator/propagator.hpp>
#include <detray/test/utils/simulation/event_generator/track_generators.hpp>

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
TEST_P(CudaCkfCombinatoricsTelescopeTests, Run) {

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

    auto field =
        detray::bfield::create_const_field<host_detector_type::scalar_type>(
            std::get<13>(GetParam()));

    // Detector view object
    auto det_view = detray::get_data(host_det);

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
        ptc, n_events, host_det, field, std::move(generator),
        std::move(smearer_writer_cfg), full_path);
    sim.run();

    /*****************************
     * Do the reconstruction
     *****************************/

    // Stream object
    traccc::cuda::stream stream;

    // Copy objects
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    traccc::device::container_d2h_copy_alg<
        traccc::track_candidate_container_types>
        track_candidate_d2h{mr, copy};

    traccc::device::container_d2h_copy_alg<traccc::track_state_container_types>
        track_state_d2h{mr, copy};

    // Seed generator
    seed_generator<host_detector_type> sg(host_det, stddevs);

    // Finding algorithm configuration
    typename traccc::cuda::finding_algorithm<
        rk_stepper_type, device_navigator_type>::config_type cfg_no_limit;
    cfg_no_limit.ptc_hypothesis = ptc;
    cfg_no_limit.max_num_branches_per_seed = 100000;
    cfg_no_limit.chi2_max = 30.f;

    typename traccc::cuda::finding_algorithm<
        rk_stepper_type, device_navigator_type>::config_type cfg_limit;
    cfg_limit.ptc_hypothesis = ptc;
    cfg_limit.max_num_branches_per_seed = 500;
    cfg_limit.chi2_max = 30.f;

    // Finding algorithm object
    traccc::cuda::finding_algorithm<rk_stepper_type, device_navigator_type>
        device_finding(cfg_no_limit, mr, copy, stream);
    traccc::cuda::finding_algorithm<rk_stepper_type, device_navigator_type>
        device_finding_limit(cfg_limit, mr, copy, stream);

    // Iterate over events
    for (std::size_t i_evt = 0; i_evt < n_events; i_evt++) {

        // Truth Track Candidates
        traccc::event_data evt_data(path, i_evt, host_mr);

        traccc::track_candidate_container_types::host truth_track_candidates =
            evt_data.generate_truth_candidates(sg, host_mr);

        ASSERT_EQ(truth_track_candidates.size(), n_truth_tracks);

        // Prepare truth seeds
        traccc::bound_track_parameters_collection_types::host seeds(&host_mr);
        for (unsigned int i_trk = 0; i_trk < n_truth_tracks; i_trk++) {
            seeds.push_back(
                truth_track_candidates.at(i_trk).header.seed_params);
        }
        ASSERT_EQ(seeds.size(), n_truth_tracks);

        traccc::bound_track_parameters_collection_types::buffer seeds_buffer{
            static_cast<unsigned int>(seeds.size()), mr.main};
        copy.setup(seeds_buffer)->wait();
        copy(vecmem::get_data(seeds), seeds_buffer,
             vecmem::copy::type::host_to_device)
            ->wait();

        // Read measurements
        traccc::measurement_collection_types::host measurements_per_event{
            &host_mr};
        traccc::io::read_measurements(measurements_per_event, i_evt, path);

        traccc::measurement_collection_types::buffer measurements_buffer(
            static_cast<unsigned int>(measurements_per_event.size()), mr.main);
        copy.setup(measurements_buffer)->wait();
        copy(vecmem::get_data(measurements_per_event), measurements_buffer)
            ->wait();

        // Instantiate output cuda containers/collections
        traccc::track_candidate_container_types::buffer
            track_candidates_cuda_buffer{{{}, *(mr.host)},
                                         {{}, *(mr.host), mr.host}};
        copy.setup(track_candidates_cuda_buffer.headers)->wait();
        copy.setup(track_candidates_cuda_buffer.items)->wait();

        traccc::track_candidate_container_types::buffer
            track_candidates_limit_cuda_buffer{{{}, *(mr.host)},
                                               {{}, *(mr.host), mr.host}};
        copy.setup(track_candidates_limit_cuda_buffer.headers)->wait();
        copy.setup(track_candidates_limit_cuda_buffer.items)->wait();

        // Run device finding
        track_candidates_cuda_buffer =
            device_finding(det_view, field, measurements_buffer, seeds_buffer);

        // Run device finding (Limit)
        track_candidates_limit_cuda_buffer = device_finding_limit(
            det_view, field, measurements_buffer, seeds_buffer);

        traccc::track_candidate_container_types::host track_candidates_cuda =
            track_candidate_d2h(track_candidates_cuda_buffer);
        traccc::track_candidate_container_types::host
            track_candidates_limit_cuda =
                track_candidate_d2h(track_candidates_limit_cuda_buffer);

        // Make sure that the number of found tracks = n_track ^ (n_planes + 1)
        ASSERT_TRUE(track_candidates_cuda.size() >
                    track_candidates_limit_cuda.size());
        ASSERT_EQ(track_candidates_cuda.size(),
                  std::pow(n_truth_tracks, std::get<11>(GetParam()) + 1));
        ASSERT_EQ(track_candidates_limit_cuda.size(),
                  n_truth_tracks * cfg_limit.max_num_branches_per_seed);
    }
}

// Testing two identical tracks
INSTANTIATE_TEST_SUITE_P(
    CUDACkfCombinatoricsTelescopeValidation, CudaCkfCombinatoricsTelescopeTests,
    ::testing::Values(
        std::make_tuple("telescope_combinatorics_twin",
                        std::array<scalar, 3u>{0.f, 0.f, 0.f},
                        std::array<scalar, 3u>{0.f, 0.f, 0.f},
                        std::array<scalar, 2u>{100.f, 100.f},
                        std::array<scalar, 2u>{0.f, 0.f},
                        std::array<scalar, 2u>{0.f, 0.f},
                        detray::muon<scalar>(), 2, 1, false, 20.f, 9u, 20.f,
                        vector3{2 * traccc::unit<scalar>::T, 0, 0}),
        std::make_tuple("telescope_combinatorics_trio",
                        std::array<scalar, 3u>{0.f, 0.f, 0.f},
                        std::array<scalar, 3u>{0.f, 0.f, 0.f},
                        std::array<scalar, 2u>{100.f, 100.f},
                        std::array<scalar, 2u>{0.f, 0.f},
                        std::array<scalar, 2u>{0.f, 0.f},
                        detray::muon<scalar>(), 3, 1, false, 20.f, 9u, 20.f,
                        vector3{2 * traccc::unit<scalar>::T, 0, 0})));
