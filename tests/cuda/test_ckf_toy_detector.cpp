/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/cuda/finding/finding_algorithm.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"
#include "traccc/device/container_h2d_copy_alg.hpp"
#include "traccc/finding/finding_algorithm.hpp"
#include "traccc/io/event_map2.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/performance/container_comparator.hpp"
#include "traccc/simulation/simulator.hpp"
#include "traccc/utils/ranges.hpp"

// Test include(s).
#include "tests/ckf_toy_detector_test.hpp"
#include "traccc/utils/seed_generator.hpp"

// detray include(s).
#include "detray/io/frontend/detector_reader.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/simulation/event_generator/track_generators.hpp"

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

TEST_P(CkfToyDetectorTests, Run) {

    // Get the parameters
    const std::string name = std::get<0>(GetParam());
    const detray::pdg_particle<scalar> ptc = std::get<6>(GetParam());
    const unsigned int n_truth_tracks = std::get<7>(GetParam());
    const unsigned int n_events = std::get<8>(GetParam());
    const bool random_charge = std::get<9>(GetParam());

    /*****************************
     * Build a toy detector
     *****************************/

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &host_mr};
    vecmem::cuda::managed_memory_resource mng_mr;

    // Read back detector file
    const std::string path = name + "/";
    detray::io::detector_reader_config reader_cfg{};
    reader_cfg.add_file(path + "toy_detector_geometry.json")
        .add_file(path + "toy_detector_homogeneous_material.json")
        .add_file(path + "toy_detector_surface_grids.json");

    auto [host_det, names] =
        detray::io::read_detector<host_detector_type>(mng_mr, reader_cfg);

    auto field = detray::bfield::create_const_field(B);

    // Detector view object
    auto det_view = detray::get_data(host_det);

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
    sim.get_config().propagation.stepping.step_constraint = step_constraint;
    sim.get_config().propagation.navigation.search_window = search_window;
    sim.run();

    /*****************************
     * Do the reconstruction
     *****************************/

    // Stream object
    traccc::cuda::stream stream;

    // Copy objects
    vecmem::cuda::async_copy copy{stream.cudaStream()};

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
    cfg.ptc_hypothesis = ptc;
    cfg.max_num_branches_per_seed = 500;
    cfg.navigation_buffer_size_scaler = 100;

    cfg.propagation.navigation.search_window = search_window;

    // Finding algorithm object
    traccc::finding_algorithm<rk_stepper_type, host_navigator_type>
        host_finding(cfg);

    // Finding algorithm object
    traccc::cuda::finding_algorithm<rk_stepper_type, device_navigator_type>
        device_finding(cfg, mr, copy, stream);

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

        traccc::bound_track_parameters_collection_types::buffer seeds_buffer{
            static_cast<unsigned int>(seeds.size()), mr.main};
        copy.setup(seeds_buffer);
        copy(vecmem::get_data(seeds), seeds_buffer,
             vecmem::copy::type::host_to_device);

        // Read measurements
        traccc::io::measurement_reader_output readOut(&host_mr);
        traccc::io::read_measurements(readOut, i_evt, path,
                                      traccc::data_format::csv);
        traccc::measurement_collection_types::host& measurements_per_event =
            readOut.measurements;

        traccc::measurement_collection_types::buffer measurements_buffer(
            measurements_per_event.size(), mr.main);
        copy(vecmem::get_data(measurements_per_event), measurements_buffer);

        // Instantiate output cuda containers/collections
        traccc::track_candidate_container_types::buffer
            track_candidates_cuda_buffer{{{}, *(mr.host)},
                                         {{}, *(mr.host), mr.host}};
        copy.setup(track_candidates_cuda_buffer.headers);
        copy.setup(track_candidates_cuda_buffer.items);

        // Navigation buffer
        auto navigation_buffer = detray::create_candidates_buffer(
            host_det,
            device_finding.get_config().navigation_buffer_size_scaler *
                seeds.size(),
            mr.main, mr.host);

        // Run host finding
        auto track_candidates =
            host_finding(host_det, field, measurements_per_event, seeds);

        // Run device finding
        track_candidates_cuda_buffer =
            device_finding(det_view, field, navigation_buffer,
                           measurements_buffer, seeds_buffer);

        traccc::track_candidate_container_types::host track_candidates_cuda =
            track_candidate_d2h(track_candidates_cuda_buffer);

        // Simple check
        ASSERT_NEAR(track_candidates.size(), track_candidates_cuda.size(), 1u);
        ASSERT_GE(track_candidates.size(), n_truth_tracks);

        // Make sure that the outputs from cpu and cuda CKF are equivalent
        unsigned int n_matches = 0u;
        for (unsigned int i = 0u; i < track_candidates.size(); i++) {
            auto iso =
                traccc::details::is_same_object(track_candidates.at(i).items);

            for (unsigned int j = 0u; j < track_candidates_cuda.size(); j++) {
                if (iso(track_candidates_cuda.at(j).items)) {
                    n_matches++;
                    break;
                }
            }
        }

        float matching_rate =
            float(n_matches) /
            std::max(track_candidates.size(), track_candidates_cuda.size());
        EXPECT_GE(matching_rate, 0.999f);
    }
}

INSTANTIATE_TEST_SUITE_P(
    CUDACkfToyDetectorValidation, CkfToyDetectorTests,
    ::testing::Values(
        std::make_tuple("toy_n_particles_1",
                        std::array<scalar, 3u>{0.f, 0.f, 0.f},
                        std::array<scalar, 3u>{0.f, 0.f, 0.f},
                        std::array<scalar, 2u>{1.f, 100.f},
                        std::array<scalar, 2u>{-4.f, 4.f},
                        std::array<scalar, 2u>{-detray::constant<scalar>::pi,
                                               detray::constant<scalar>::pi},
                        detray::muon<scalar>(), 1, 1, false),
        std::make_tuple("toy_n_particles_10000",
                        std::array<scalar, 3u>{0.f, 0.f, 0.f},
                        std::array<scalar, 3u>{0.f, 0.f, 0.f},
                        std::array<scalar, 2u>{1.f, 100.f},
                        std::array<scalar, 2u>{-4.f, 4.f},
                        std::array<scalar, 2u>{-detray::constant<scalar>::pi,
                                               detray::constant<scalar>::pi},
                        detray::muon<scalar>(), 10000, 1, false),
        std::make_tuple("toy_n_particles_10000_random_charge",
                        std::array<scalar, 3u>{0.f, 0.f, 0.f},
                        std::array<scalar, 3u>{0.f, 0.f, 0.f},
                        std::array<scalar, 2u>{1.f, 100.f},
                        std::array<scalar, 2u>{-4.f, 4.f},
                        std::array<scalar, 2u>{-detray::constant<scalar>::pi,
                                               detray::constant<scalar>::pi},
                        detray::muon<scalar>(), 10000, 1, true)));
