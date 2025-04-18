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
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <chrono>
#include <random>

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

bool find_pattern(
    const track_candidate_container_types::device& res_track_candidates,
    const std::vector<std::size_t>& pattern) {

    unsigned int n_tracks = res_track_candidates.size();
    for (unsigned i = 0; i < n_tracks; i++) {
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
    fill_pattern(trk_cands, 0, 8.3f, {1, 3, 5, 11});
    fill_pattern(trk_cands, 1, 2.2f, {6, 7, 8, 9, 10, 12});
    fill_pattern(trk_cands, 2, 3.7f, {2, 4, 13});

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
                                 std::vector<std::size_t>({1, 3, 5, 11})));
        ASSERT_TRUE(find_pattern(
            res_trk_cands, std::vector<std::size_t>({6, 7, 8, 9, 10, 12})));
        ASSERT_TRUE(
            find_pattern(res_trk_cands, std::vector<std::size_t>({2, 4, 13})));
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
                  std::vector<std::size_t>({6, 7, 8, 9, 10, 12}));
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
    fill_pattern(trk_cands, 0, 0.12f, {1, 3, 5, 11, 14, 16, 18});
    fill_pattern(trk_cands, 1, 0.53f, {3, 5, 6, 13});

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
        res_trk_cands, std::vector<std::size_t>({1, 3, 5, 11, 14, 16, 18})));
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
    fill_pattern(trk_cands, 0, 0.2f, {1, 3, 5, 11});
    fill_pattern(trk_cands, 1, 0.5f, {2, 6});
    fill_pattern(trk_cands, 2, 0.4f, {3, 6, 12, 14, 19, 21});
    fill_pattern(trk_cands, 3, 0.1f, {2, 7, 11, 13, 16});
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
                             std::vector<std::size_t>({3, 6, 12, 14, 19, 21})));
    ASSERT_TRUE(find_pattern(res_trk_cands,
                             std::vector<std::size_t>({2, 7, 11, 13, 16})));
}

// Comparison to the CPU algorithm
TEST(CudaAmbiguitySolverTests, GreedyResolverTest4) {

    std::size_t n_tracks = 10000u;

    // Memory resource used by the EDM.
    vecmem::cuda::managed_memory_resource mng_mr;
    vecmem::host_memory_resource host_mr;
    traccc::memory_resource mr{mng_mr, &host_mr};

    // Cuda stream
    traccc::cuda::stream stream;

    // Cuda copy objects
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    track_candidate_container_types::host trk_cands{&mr.main};

    trk_cands.resize(n_tracks);
    std::mt19937 gen(42);

    for (std::size_t i = 0; i < n_tracks; i++) {

        std::uniform_int_distribution<std::size_t> track_length_dist(1, 20);
        std::uniform_int_distribution<std::size_t> meas_id_dist(0, 10000);
        std::uniform_real_distribution<traccc::scalar> pval_dist(0.0f, 1.0f);

        const std::size_t track_length = track_length_dist(gen);
        const traccc::scalar pval = pval_dist(gen);
        std::vector<std::size_t> pattern;
        while (pattern.size() < track_length) {

            const std::size_t meas_id = meas_id_dist(gen);
            if (std::find(pattern.begin(), pattern.end(), meas_id) ==
                pattern.end()) {
                pattern.push_back(meas_id);
            }
        }

        std::sort(pattern.begin(), pattern.end());
        auto last = std::unique(pattern.begin(), pattern.end());

        // There should not be duplicate
        ASSERT_EQ(last, pattern.end());
        pattern.erase(last, pattern.end());

        // Make sure that partern size is eqaul to the track length
        ASSERT_EQ(pattern.size(), track_length);

        // Fill the pattern
        fill_pattern(trk_cands, i, pval, pattern);
    }

    // CPU algorithm
    traccc::host::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::host::greedy_ambiguity_resolution_algorithm resolution_alg_cpu(
        resolution_config, mr);

    auto start_cpu = std::chrono::high_resolution_clock::now();

    auto res_trk_cands_cpu = resolution_alg_cpu(traccc::get_data(trk_cands));

    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_cpu - start_cpu);
    std::cout << " Time for the cpu method " << duration_cpu.count() << " ms"
              << std::endl;

    // CUDA algorithm
    traccc::cuda::greedy_ambiguity_resolution_algorithm resolution_alg_cuda(
        resolution_config, mr, copy, stream);

    auto start_cuda = std::chrono::high_resolution_clock::now();

    auto res_trk_cands_buffer =
        resolution_alg_cuda(traccc::get_data(trk_cands));
    track_candidate_container_types::device res_trk_cands_cuda(
        res_trk_cands_buffer);

    auto end_cuda = std::chrono::high_resolution_clock::now();
    auto duration_cuda = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_cuda - start_cuda);
    std::cout << " Time for the cuda method " << duration_cuda.count() << " ms"
              << std::endl;

    const auto n_tracks_cpu = res_trk_cands_cpu.size();
    ASSERT_EQ(n_tracks_cpu, res_trk_cands_cuda.size());

    // Make sure that CPU and CUDA track candidates have same patterns
    for (unsigned int i = 0; i < n_tracks_cpu; i++) {
        ASSERT_TRUE(find_pattern(res_trk_cands_cuda,
                                 get_pattern(res_trk_cands_cpu, i)));
    }
}
