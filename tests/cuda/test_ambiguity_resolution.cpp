/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/ambiguity_resolution/ambiguity_resolution_config.hpp"
#include "traccc/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/cuda/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"
#include "traccc/device/container_h2d_copy_alg.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <chrono>
#include <random>
#include <thread>

using namespace traccc;

void fill_pattern(track_candidate_container_types::host& track_candidates,
                  const std::size_t idx, const traccc::scalar pval,
                  const std::vector<std::size_t>& pattern) {

    track_candidates.at(idx).header.trk_quality.pval = pval;

    auto& cands = track_candidates.at(idx).items;
    for (const auto& meas_id : pattern) {
        cands.push_back({});
        cands.back().measurement_id = meas_id;
    }
}

template <typename track_container_t>
bool find_pattern(const track_container_t& res_track_candidates,
                  const std::vector<std::size_t>& pattern) {

    const auto n_tracks = res_track_candidates.size();
    for (unsigned int i = 0; i < n_tracks; i++) {
        const auto& res_cands = res_track_candidates.at(i).items;
        std::vector<std::size_t> ids;
        for (const auto& cand : res_cands) {
            ids.push_back(cand.measurement_id);
        }
        if (pattern == ids) {
            return true;
        }
    }
    return false;
}

template <typename track_candidate_container_t>
std::vector<std::size_t> get_pattern(
    const track_candidate_container_t& track_candidates,
    const typename track_candidate_container_t::size_type idx) {
    std::vector<std::size_t> ret;
    const auto& cands = track_candidates.at(idx).items;
    for (const auto& cand : cands) {
        ret.push_back(cand.measurement_id);
    }

    return ret;
}

TEST(CudaAmbiguitySolverTests, GreedyResolverTest0) {

    // Memory resource used by the EDM.
    vecmem::cuda::managed_memory_resource mng_mr;
    vecmem::host_memory_resource host_mr;
    traccc::memory_resource mr{mng_mr, &host_mr};

    // Cuda stream
    traccc::cuda::stream stream;

    // Cuda copy objects
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    track_candidate_container_types::host trk_cands{&mr.main};

    trk_cands.resize(3u);
    fill_pattern(trk_cands, 0, 0.23f, {5, 1, 11, 3});
    fill_pattern(trk_cands, 1, 0.85f, {12, 10, 9, 8, 7, 6});
    fill_pattern(trk_cands, 2, 0.42f, {4, 2, 13});

    traccc::cuda::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;

    traccc::cuda::greedy_ambiguity_resolution_algorithm resolution_alg_cuda(
        resolution_config, mr, copy, stream);
    {
        resolution_alg_cuda.get_config().min_meas_per_track = 3;
        auto res_trk_cands_buffer =
            resolution_alg_cuda(traccc::get_data(trk_cands));
        track_candidate_container_types::device res_trk_cands(
            res_trk_cands_buffer);
        // All tracks are accepted as they have more than three measurements
        EXPECT_EQ(res_trk_cands.size(), 3u);
        ASSERT_TRUE(find_pattern(res_trk_cands,
                                 std::vector<std::size_t>({5, 1, 11, 3})));
        ASSERT_TRUE(find_pattern(
            res_trk_cands, std::vector<std::size_t>({12, 10, 9, 8, 7, 6})));
        ASSERT_TRUE(
            find_pattern(res_trk_cands, std::vector<std::size_t>({4, 2, 13})));
    }

    {
        resolution_alg_cuda.get_config().min_meas_per_track = 5;
        auto res_trk_cands_buffer =
            resolution_alg_cuda(traccc::get_data(trk_cands));
        track_candidate_container_types::device res_trk_cands(
            res_trk_cands_buffer);
        // Only the second track with six measurements is accepted
        EXPECT_EQ(res_trk_cands.size(), 1u);
        EXPECT_EQ(get_pattern(res_trk_cands, 0),
                  std::vector<std::size_t>({12, 10, 9, 8, 7, 6}));
    }
}

TEST(CudaAmbiguitySolverTests, GreedyResolverTest1) {

    // Memory resource used by the EDM.
    vecmem::cuda::managed_memory_resource mng_mr;
    vecmem::host_memory_resource host_mr;
    traccc::memory_resource mr{mng_mr, &host_mr};

    // Cuda stream
    traccc::cuda::stream stream;

    // Cuda copy objects
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    track_candidate_container_types::host trk_cands{&mr.main};

    trk_cands.resize(2u);
    fill_pattern(trk_cands, 0, 0.12f, {5, 14, 1, 11, 18, 16, 3});
    fill_pattern(trk_cands, 1, 0.53f, {3, 6, 5, 13});

    traccc::cuda::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;

    traccc::cuda::greedy_ambiguity_resolution_algorithm resolution_alg_cuda(
        resolution_config, mr, copy, stream);

    resolution_alg_cuda.get_config().min_meas_per_track = 3;
    auto res_trk_cands_buffer =
        resolution_alg_cuda(traccc::get_data(trk_cands));
    track_candidate_container_types::device res_trk_cands(res_trk_cands_buffer);
    // All tracks are accepted as they have more than three measurements
    ASSERT_EQ(res_trk_cands.size(), 1u);

    // The first track is selected over the second one as its relative
    // shared measurement (2/7) is lower than the one of the second track
    // (2/4)
    ASSERT_TRUE(find_pattern(
        res_trk_cands, std::vector<std::size_t>({5, 14, 1, 11, 18, 16, 3})));
}

TEST(CudaAmbiguitySolverTests, GreedyResolverTest2) {

    // Memory resource used by the EDM.
    vecmem::cuda::managed_memory_resource mng_mr;
    vecmem::host_memory_resource host_mr;
    traccc::memory_resource mr{mng_mr, &host_mr};

    // Cuda stream
    traccc::cuda::stream stream;

    // Cuda copy objects
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    track_candidate_container_types::host trk_cands{&mr.main};

    trk_cands.resize(2u);
    fill_pattern(trk_cands, 0, 0.8f, {1, 3, 5, 11});
    fill_pattern(trk_cands, 1, 0.9f, {3, 5, 6, 13});

    traccc::cuda::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::cuda::greedy_ambiguity_resolution_algorithm resolution_alg_cuda(
        resolution_config, mr, copy, stream);

    auto res_trk_cands_buffer =
        resolution_alg_cuda(traccc::get_data(trk_cands));
    track_candidate_container_types::device res_trk_cands(res_trk_cands_buffer);
    ASSERT_EQ(res_trk_cands.size(), 1u);

    // The second track is selected over the first one as their relative
    // shared measurement (2/4) is the same but its p-value is higher
    ASSERT_TRUE(
        find_pattern(res_trk_cands, std::vector<std::size_t>({3, 5, 6, 13})));
}

TEST(CudaAmbiguitySolverTests, GreedyResolverTest3) {

    // Memory resource used by the EDM.
    vecmem::cuda::managed_memory_resource mng_mr;
    vecmem::host_memory_resource host_mr;
    traccc::memory_resource mr{mng_mr, &host_mr};

    // Cuda stream
    traccc::cuda::stream stream;

    // Cuda copy objects
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    track_candidate_container_types::host trk_cands{&mr.main};

    trk_cands.resize(6u);

    fill_pattern(trk_cands, 0, 0.2f, {5, 1, 11, 3});
    fill_pattern(trk_cands, 1, 0.5f, {6, 2});
    fill_pattern(trk_cands, 2, 0.4f, {3, 21, 12, 6, 19, 14});
    fill_pattern(trk_cands, 3, 0.1f, {13, 16, 2, 7, 11});
    fill_pattern(trk_cands, 4, 0.3f, {1, 7, 8});
    fill_pattern(trk_cands, 5, 0.6f, {1, 3, 11, 22});

    traccc::cuda::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::cuda::greedy_ambiguity_resolution_algorithm resolution_alg_cuda(
        resolution_config, mr, copy, stream);

    auto res_trk_cands_buffer =
        resolution_alg_cuda(traccc::get_data(trk_cands));
    track_candidate_container_types::device res_trk_cands(res_trk_cands_buffer);
    ASSERT_EQ(res_trk_cands.size(), 2u);

    ASSERT_TRUE(find_pattern(res_trk_cands,
                             std::vector<std::size_t>({3, 21, 12, 6, 19, 14})));
    ASSERT_TRUE(find_pattern(res_trk_cands,
                             std::vector<std::size_t>({13, 16, 2, 7, 11})));
}

// Comparison to the CPU algorithm
TEST(CudaAmbiguitySolverTests, CPU_Comparison) {

    std::size_t n_tracks = 50000u;

    // Memory resource used by the EDM.
    vecmem::cuda::device_memory_resource device_mr;
    vecmem::host_memory_resource host_mr;
    traccc::memory_resource mr{device_mr, &host_mr};
    traccc::memory_resource hmr{host_mr};

    // Cuda stream
    traccc::cuda::stream stream;

    // Cuda copy objects
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    // Copy algorithms
    traccc::device::container_h2d_copy_alg<
        traccc::track_candidate_container_types>
        track_candidate_h2d{mr, copy};

    traccc::device::container_d2h_copy_alg<
        traccc::track_candidate_container_types>
        track_candidate_d2h{mr, copy};

    for (int i_evt = 0; i_evt < 5; i_evt++) {

        std::size_t sd = 42 + i_evt;
        std::mt19937 gen(sd);
        std::cout << "Event: " << i_evt << " Seed: " << sd << std::endl;

        track_candidate_container_types::host trk_cands{&host_mr};
        trk_cands.resize(n_tracks);

        for (std::size_t i = 0; i < n_tracks; i++) {

            std::uniform_int_distribution<std::size_t> track_length_dist(1, 10);
            std::uniform_int_distribution<std::size_t> meas_id_dist(0, 20000);
            std::uniform_real_distribution<traccc::scalar> pval_dist(0.0f,
                                                                     1.0f);

            const std::size_t track_length = track_length_dist(gen);
            const traccc::scalar pval = pval_dist(gen);
            std::vector<std::size_t> pattern;
            // std::cout << pval << std::endl;
            while (pattern.size() < track_length) {
                const auto mid = meas_id_dist(gen);
                // std::cout << mid << ", ";
                pattern.push_back(mid);
            }
            // std::cout << std::endl;

            // Make sure that partern size is eqaul to the track length
            ASSERT_EQ(pattern.size(), track_length);

            // Fill the pattern
            fill_pattern(trk_cands, i, pval, pattern);
        }

        // CPU algorithm
        traccc::host::greedy_ambiguity_resolution_algorithm::config_type
            resolution_config;
        traccc::host::greedy_ambiguity_resolution_algorithm resolution_alg_cpu(
            resolution_config, hmr);

        auto start_cpu = std::chrono::high_resolution_clock::now();

        auto res_trk_cands_cpu =
            resolution_alg_cpu(traccc::get_data(trk_cands));

        auto end_cpu = std::chrono::high_resolution_clock::now();
        auto duration_cpu =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu -
                                                                  start_cpu);
        std::cout << " Time for the cpu method " << duration_cpu.count()
                  << " ms" << std::endl;

        // CUDA algorithm
        traccc::cuda::greedy_ambiguity_resolution_algorithm resolution_alg_cuda(
            resolution_config, mr, copy, stream);

        // H2D transfer
        traccc::track_candidate_container_types::buffer trk_cands_buffer =
            track_candidate_h2d(traccc::get_data(trk_cands));
        copy.setup(trk_cands_buffer.headers)->wait();
        copy.setup(trk_cands_buffer.items)->wait();

        auto start_cuda = std::chrono::high_resolution_clock::now();

        // Instantiate output cuda containers/collections
        traccc::track_candidate_container_types::buffer res_trk_cands_buffer{
            {{}, *(mr.host)}, {{}, *(mr.host), mr.host}};
        copy.setup(res_trk_cands_buffer.headers)->wait();
        copy.setup(res_trk_cands_buffer.items)->wait();

        res_trk_cands_buffer = resolution_alg_cuda(trk_cands_buffer);

        auto end_cuda = std::chrono::high_resolution_clock::now();
        auto duration_cuda =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_cuda -
                                                                  start_cuda);
        std::cout << " Time for the cuda method " << duration_cuda.count()
                  << " ms" << std::endl;

        traccc::track_candidate_container_types::host res_trk_cands_cuda =
            track_candidate_d2h(res_trk_cands_buffer);

        const auto n_tracks_cpu = res_trk_cands_cpu.size();
        ASSERT_EQ(n_tracks_cpu, res_trk_cands_cuda.size());

        // Make sure that CPU and CUDA track candidates have same patterns
        for (unsigned int i = 0; i < n_tracks_cpu; i++) {
            ASSERT_TRUE(find_pattern(res_trk_cands_cuda,
                                     get_pattern(res_trk_cands_cpu, i)));
        }
    }
}

TEST(CudaAmbiguitySolverTests, GreedyResolverTest5) {

    // Memory resource used by the EDM.
    vecmem::cuda::managed_memory_resource mng_mr;
    vecmem::host_memory_resource host_mr;
    traccc::memory_resource mr{mng_mr, &host_mr};

    // Cuda stream
    traccc::cuda::stream stream;

    // Cuda copy objects
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    track_candidate_container_types::host trk_cands{&mr.main};

    trk_cands.resize(4u);
    fill_pattern(trk_cands, 0, 0.2f, {1, 2, 1, 1});
    fill_pattern(trk_cands, 1, 0.5f, {3, 2, 1});
    fill_pattern(trk_cands, 2, 0.4f, {2, 4, 5, 7, 2});
    fill_pattern(trk_cands, 3, 0.1f, {6, 6, 6, 6});

    traccc::cuda::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::cuda::greedy_ambiguity_resolution_algorithm resolution_alg_cuda(
        resolution_config, mr, copy, stream);

    auto res_trk_cands_buffer =
        resolution_alg_cuda(traccc::get_data(trk_cands));
    track_candidate_container_types::device res_trk_cands(res_trk_cands_buffer);
    ASSERT_EQ(res_trk_cands.size(), 2u);

    ASSERT_TRUE(
        find_pattern(res_trk_cands, std::vector<std::size_t>({3, 2, 1})));
    ASSERT_TRUE(
        find_pattern(res_trk_cands, std::vector<std::size_t>({6, 6, 6, 6})));
}

TEST(CudaAmbiguitySolverTests, GreedyResolverTest6) {

    // Memory resource used by the EDM.
    vecmem::cuda::managed_memory_resource mng_mr;
    vecmem::host_memory_resource host_mr;
    traccc::memory_resource mr{mng_mr, &host_mr};

    // Cuda stream
    traccc::cuda::stream stream;

    // Cuda copy objects
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    track_candidate_container_types::host trk_cands{&mr.main};

    trk_cands.resize(5u);
    fill_pattern(trk_cands, 0, 0.2f, {7, 3, 5, 7, 7, 7, 2});
    fill_pattern(trk_cands, 1, 0.5f, {2});
    fill_pattern(trk_cands, 2, 0.4f, {8, 9, 7, 2, 3, 4, 3, 7});
    fill_pattern(trk_cands, 3, 0.1f, {8, 9, 0, 8, 1, 4, 6});
    fill_pattern(trk_cands, 4, 0.9f, {10, 3, 2});

    traccc::cuda::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::cuda::greedy_ambiguity_resolution_algorithm resolution_alg_cuda(
        resolution_config, mr, copy, stream);

    auto res_trk_cands_buffer =
        resolution_alg_cuda(traccc::get_data(trk_cands));
    track_candidate_container_types::device res_trk_cands(res_trk_cands_buffer);
    ASSERT_EQ(res_trk_cands.size(), 2u);

    ASSERT_TRUE(find_pattern(res_trk_cands,
                             std::vector<std::size_t>({7, 3, 5, 7, 7, 7, 2})));
    ASSERT_TRUE(find_pattern(res_trk_cands,
                             std::vector<std::size_t>({8, 9, 0, 8, 1, 4, 6})));
}

TEST(CudaAmbiguitySolverTests, GreedyResolverTest7) {

    // Memory resource used by the EDM.
    vecmem::cuda::managed_memory_resource mng_mr;
    vecmem::host_memory_resource host_mr;
    traccc::memory_resource mr{mng_mr, &host_mr};

    // Cuda stream
    traccc::cuda::stream stream;

    // Cuda copy objects
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    track_candidate_container_types::host trk_cands{&mr.main};

    trk_cands.resize(3u);
    fill_pattern(trk_cands, 0, 0.173853f, {10, 3, 6, 8});
    fill_pattern(trk_cands, 1, 0.548019f, {3, 3, 1});
    fill_pattern(trk_cands, 2, 0.276757f, {2, 8, 5, 4});

    traccc::cuda::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::cuda::greedy_ambiguity_resolution_algorithm resolution_alg_cuda(
        resolution_config, mr, copy, stream);

    auto res_trk_cands_buffer =
        resolution_alg_cuda(traccc::get_data(trk_cands));
    track_candidate_container_types::device res_trk_cands(res_trk_cands_buffer);
    ASSERT_EQ(res_trk_cands.size(), 1u);

    ASSERT_TRUE(
        find_pattern(res_trk_cands, std::vector<std::size_t>({2, 8, 5, 4})));
}

TEST(CudaAmbiguitySolverTests, GreedyResolverTest8) {

    // Memory resource used by the EDM.
    vecmem::cuda::managed_memory_resource mng_mr;
    vecmem::host_memory_resource host_mr;
    traccc::memory_resource mr{mng_mr, &host_mr};

    // Cuda stream
    traccc::cuda::stream stream;

    // Cuda copy objects
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    track_candidate_container_types::host trk_cands{&mr.main};

    trk_cands.resize(5u);
    fill_pattern(trk_cands, 0, 0.0623132f, {10, 4});
    fill_pattern(trk_cands, 1, 0.207417f, {6, 7, 5});
    fill_pattern(trk_cands, 2, 0.325736f, {8, 2, 2});
    fill_pattern(trk_cands, 3, 0.581643f, {5, 7, 9, 7});
    fill_pattern(trk_cands, 4, 0.389551f, {1, 9, 3, 0});

    traccc::cuda::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::cuda::greedy_ambiguity_resolution_algorithm resolution_alg_cuda(
        resolution_config, mr, copy, stream);

    auto res_trk_cands_buffer =
        resolution_alg_cuda(traccc::get_data(trk_cands));
    track_candidate_container_types::device res_trk_cands(res_trk_cands_buffer);
    ASSERT_EQ(res_trk_cands.size(), 3u);

    ASSERT_TRUE(
        find_pattern(res_trk_cands, std::vector<std::size_t>({6, 7, 5})));
    ASSERT_TRUE(
        find_pattern(res_trk_cands, std::vector<std::size_t>({8, 2, 2})));
    ASSERT_TRUE(
        find_pattern(res_trk_cands, std::vector<std::size_t>({1, 9, 3, 0})));
}

TEST(CudaAmbiguitySolverTests, GreedyResolverTest9) {

    // Memory resource used by the EDM.
    vecmem::cuda::managed_memory_resource mng_mr;
    vecmem::host_memory_resource host_mr;
    traccc::memory_resource mr{mng_mr, &host_mr};

    // Cuda stream
    traccc::cuda::stream stream;

    // Cuda copy objects
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    track_candidate_container_types::host trk_cands{&mr.main};

    trk_cands.resize(3u);
    fill_pattern(trk_cands, 0, 0.542984f, {0, 4, 8, 1, 1});
    fill_pattern(trk_cands, 1, 0.583695f, {10, 6, 8, 7});
    fill_pattern(trk_cands, 2, 0.280232f, {4, 1, 8, 10});

    traccc::cuda::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::cuda::greedy_ambiguity_resolution_algorithm resolution_alg_cuda(
        resolution_config, mr, copy, stream);

    auto res_trk_cands_buffer =
        resolution_alg_cuda(traccc::get_data(trk_cands));
    track_candidate_container_types::device res_trk_cands(res_trk_cands_buffer);
    ASSERT_EQ(res_trk_cands.size(), 1u);

    ASSERT_TRUE(
        find_pattern(res_trk_cands, std::vector<std::size_t>({0, 4, 8, 1, 1})));
}

TEST(CudaAmbiguitySolverTests, GreedyResolverTest10) {

    // Memory resource used by the EDM.
    vecmem::cuda::managed_memory_resource mng_mr;
    vecmem::host_memory_resource host_mr;
    traccc::memory_resource mr{mng_mr, &host_mr};

    // Cuda stream
    traccc::cuda::stream stream;

    // Cuda copy objects
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    track_candidate_container_types::host trk_cands{&mr.main};

    trk_cands.resize(5u);
    fill_pattern(trk_cands, 0, 0.399106f, {14, 51});
    fill_pattern(trk_cands, 1, 0.43899f, {80, 35, 41, 55});
    fill_pattern(trk_cands, 2, 0.0954247f, {73, 63, 49, 89});
    fill_pattern(trk_cands, 3, 0.158046f, {81, 22, 58, 54, 91});
    fill_pattern(trk_cands, 4, 0.349878f, {97, 89, 80});

    traccc::cuda::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::cuda::greedy_ambiguity_resolution_algorithm resolution_alg_cuda(
        resolution_config, mr, copy, stream);

    auto res_trk_cands_buffer =
        resolution_alg_cuda(traccc::get_data(trk_cands));
    track_candidate_container_types::device res_trk_cands(res_trk_cands_buffer);
    ASSERT_EQ(res_trk_cands.size(), 3u);

    ASSERT_TRUE(find_pattern(res_trk_cands,
                             std::vector<std::size_t>({80, 35, 41, 55})));
    ASSERT_TRUE(find_pattern(res_trk_cands,
                             std::vector<std::size_t>({73, 63, 49, 89})));
    ASSERT_TRUE(find_pattern(res_trk_cands,
                             std::vector<std::size_t>({81, 22, 58, 54, 91})));
}

TEST(CudaAmbiguitySolverTests, GreedyResolverTest11) {

    // Memory resource used by the EDM.
    vecmem::cuda::managed_memory_resource mng_mr;
    vecmem::host_memory_resource host_mr;
    traccc::memory_resource mr{mng_mr, &host_mr};

    // Cuda stream
    traccc::cuda::stream stream;

    // Cuda copy objects
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    track_candidate_container_types::host trk_cands{&mr.main};

    trk_cands.resize(5u);
    fill_pattern(trk_cands, 0, 0.95f, {56, 87});
    fill_pattern(trk_cands, 1, 0.894f, {64, 63});
    fill_pattern(trk_cands, 2, 0.824f, {70, 17});
    fill_pattern(trk_cands, 3, 0.862f, {27, 0});
    fill_pattern(trk_cands, 4, 0.871f, {27, 19});

    traccc::cuda::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::cuda::greedy_ambiguity_resolution_algorithm resolution_alg_cuda(
        resolution_config, mr, copy, stream);

    auto res_trk_cands_buffer =
        resolution_alg_cuda(traccc::get_data(trk_cands));
    track_candidate_container_types::device res_trk_cands(res_trk_cands_buffer);
    ASSERT_EQ(res_trk_cands.size(), 0u);
}

TEST(CudaAmbiguitySolverTests, GreedyResolverTest12) {

    // Memory resource used by the EDM.
    vecmem::cuda::managed_memory_resource mng_mr;
    vecmem::host_memory_resource host_mr;
    traccc::memory_resource mr{mng_mr, &host_mr};

    // Cuda stream
    traccc::cuda::stream stream;

    // Cuda copy objects
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    track_candidate_container_types::host trk_cands{&mr.main};

    trk_cands.resize(5u);
    fill_pattern(trk_cands, 0, 0.948f, {17, 6, 1, 69, 78});  // 69
    fill_pattern(trk_cands, 1, 0.609f, {17});
    fill_pattern(trk_cands, 2, 0.453f, {84, 45, 81, 69});      // 84, 69
    fill_pattern(trk_cands, 3, 0.910f, {54, 64, 49, 96, 40});  // 64
    fill_pattern(trk_cands, 4, 0.153f, {59, 57, 84, 27, 64});  // 84, 64

    traccc::cuda::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::cuda::greedy_ambiguity_resolution_algorithm resolution_alg_cuda(
        resolution_config, mr, copy, stream);

    auto res_trk_cands_buffer =
        resolution_alg_cuda(traccc::get_data(trk_cands));
    track_candidate_container_types::device res_trk_cands(res_trk_cands_buffer);
    ASSERT_EQ(res_trk_cands.size(), 2u);

    ASSERT_TRUE(find_pattern(res_trk_cands,
                             std::vector<std::size_t>({17, 6, 1, 69, 78})));
    ASSERT_TRUE(find_pattern(res_trk_cands,
                             std::vector<std::size_t>({54, 64, 49, 96, 40})));
}

TEST(CudaAmbiguitySolverTests, GreedyResolverTest13) {

    // Memory resource used by the EDM.
    vecmem::cuda::managed_memory_resource mng_mr;
    vecmem::host_memory_resource host_mr;
    traccc::memory_resource mr{mng_mr, &host_mr};

    // Cuda stream
    traccc::cuda::stream stream;

    // Cuda copy objects
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    track_candidate_container_types::host trk_cands{&mr.main};

    trk_cands.resize(6u);
    fill_pattern(trk_cands, 0, 0.211f, {46, 92, 74, 58});
    fill_pattern(trk_cands, 1, 0.694f, {15, 78, 9});
    fill_pattern(trk_cands, 2, 0.432f, {15, 4, 58, 68});
    fill_pattern(trk_cands, 3, 0.958f, {38, 93, 68});
    fill_pattern(trk_cands, 4, 0.203f, {57, 64, 57, 36});
    fill_pattern(trk_cands, 5, 0.118f, {4, 85, 65, 14});

    traccc::cuda::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::cuda::greedy_ambiguity_resolution_algorithm resolution_alg_cuda(
        resolution_config, mr, copy, stream);

    auto res_trk_cands_buffer =
        resolution_alg_cuda(traccc::get_data(trk_cands));
    track_candidate_container_types::device res_trk_cands(res_trk_cands_buffer);
    ASSERT_EQ(res_trk_cands.size(), 5u);

    ASSERT_TRUE(find_pattern(res_trk_cands,
                             std::vector<std::size_t>({46, 92, 74, 58})));
    ASSERT_TRUE(
        find_pattern(res_trk_cands, std::vector<std::size_t>({15, 78, 9})));
    ASSERT_TRUE(
        find_pattern(res_trk_cands, std::vector<std::size_t>({38, 93, 68})));
    ASSERT_TRUE(find_pattern(res_trk_cands,
                             std::vector<std::size_t>({57, 64, 57, 36})));
    ASSERT_TRUE(
        find_pattern(res_trk_cands, std::vector<std::size_t>({4, 85, 65, 14})));
}

TEST(CudaAmbiguitySolverTests, GreedyResolverTest14) {

    // Memory resource used by the EDM.
    vecmem::cuda::managed_memory_resource mng_mr;
    vecmem::host_memory_resource host_mr;
    traccc::memory_resource mr{mng_mr, &host_mr};

    // Cuda stream
    traccc::cuda::stream stream;

    // Cuda copy objects
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    track_candidate_container_types::host trk_cands{&mr.main};

    trk_cands.resize(4u);
    fill_pattern(trk_cands, 0, 0.932f, {8, 4, 3});
    fill_pattern(trk_cands, 1, 0.263f, {1, 1, 9, 3});
    fill_pattern(trk_cands, 2, 0.876f, {1, 2, 5});
    fill_pattern(trk_cands, 3, 0.058f, {2, 0, 4, 7});

    traccc::cuda::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::cuda::greedy_ambiguity_resolution_algorithm resolution_alg_cuda(
        resolution_config, mr, copy, stream);

    auto res_trk_cands_buffer =
        resolution_alg_cuda(traccc::get_data(trk_cands));
    track_candidate_container_types::device res_trk_cands(res_trk_cands_buffer);
    ASSERT_EQ(res_trk_cands.size(), 2u);

    ASSERT_TRUE(
        find_pattern(res_trk_cands, std::vector<std::size_t>({8, 4, 3})));
    ASSERT_TRUE(
        find_pattern(res_trk_cands, std::vector<std::size_t>({1, 2, 5})));
}

TEST(CudaAmbiguitySolverTests, GreedyResolverTest15) {

    // Memory resource used by the EDM.
    vecmem::cuda::managed_memory_resource mng_mr;
    vecmem::host_memory_resource host_mr;
    traccc::memory_resource mr{mng_mr, &host_mr};

    // Cuda stream
    traccc::cuda::stream stream;

    // Cuda copy objects
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    track_candidate_container_types::host trk_cands{&mr.main};

    trk_cands.resize(4u);
    fill_pattern(trk_cands, 0, 0.293f, {2, 0, 4});
    fill_pattern(trk_cands, 1, 0.362f, {8, 4, 9, 3});
    fill_pattern(trk_cands, 2, 0.011f, {9, 4, 8, 4});
    fill_pattern(trk_cands, 3, 0.843f, {8, 7, 1});

    traccc::cuda::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::cuda::greedy_ambiguity_resolution_algorithm resolution_alg_cuda(
        resolution_config, mr, copy, stream);

    auto res_trk_cands_buffer =
        resolution_alg_cuda(traccc::get_data(trk_cands));
    track_candidate_container_types::device res_trk_cands(res_trk_cands_buffer);
    ASSERT_EQ(res_trk_cands.size(), 2u);

    ASSERT_TRUE(
        find_pattern(res_trk_cands, std::vector<std::size_t>({2, 0, 4})));
    ASSERT_TRUE(
        find_pattern(res_trk_cands, std::vector<std::size_t>({8, 7, 1})));
}

TEST(CudaAmbiguitySolverTests, GreedyResolverTest16) {

    // Memory resource used by the EDM.
    vecmem::cuda::managed_memory_resource mng_mr;
    vecmem::host_memory_resource host_mr;
    traccc::memory_resource mr{mng_mr, &host_mr};

    // Cuda stream
    traccc::cuda::stream stream;

    // Cuda copy objects
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    track_candidate_container_types::host trk_cands{&mr.main};

    trk_cands.resize(5u);
    fill_pattern(trk_cands, 0, 0.622598f, {95, 24, 62, 83, 67});
    fill_pattern(trk_cands, 1, 0.541774f, {6, 52, 57, 87, 75});
    fill_pattern(trk_cands, 2, 0.361033f, {14, 52, 29, 79, 89});
    fill_pattern(trk_cands, 3, 0.622598f, {57, 85, 63, 90});
    fill_pattern(trk_cands, 4, 0.481157f, {80, 45, 94});

    traccc::cuda::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::cuda::greedy_ambiguity_resolution_algorithm resolution_alg_cuda(
        resolution_config, mr, copy, stream);

    auto res_trk_cands_buffer =
        resolution_alg_cuda(traccc::get_data(trk_cands));
    track_candidate_container_types::device res_trk_cands(res_trk_cands_buffer);
    ASSERT_EQ(res_trk_cands.size(), 4u);

    ASSERT_TRUE(find_pattern(res_trk_cands,
                             std::vector<std::size_t>({95, 24, 62, 83, 67})));
    ASSERT_TRUE(find_pattern(res_trk_cands,
                             std::vector<std::size_t>({14, 52, 29, 79, 89})));
    ASSERT_TRUE(find_pattern(res_trk_cands,
                             std::vector<std::size_t>({57, 85, 63, 90})));
    ASSERT_TRUE(
        find_pattern(res_trk_cands, std::vector<std::size_t>({80, 45, 94})));
}
