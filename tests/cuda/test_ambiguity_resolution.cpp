/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/ambiguity_resolution/ambiguity_resolution_config.hpp"
#include "traccc/cuda/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

// GTest include(s).
#include <gtest/gtest.h>

using namespace traccc;

void fill_pattern(track_candidate_container_types::host& track_candidates,
                  const std::size_t idx, const traccc::scalar chi2,
                  const std::vector<std::size_t>& pattern) {

    track_candidates.at(idx).header.trk_quality.chi2 = chi2;

    auto& cands = track_candidates.at(idx).items;
    for (const auto& meas_id : pattern) {
        cands.push_back({});
        cands.back().measurement_id = meas_id;
    }
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
    {
        resolution_alg_cuda.get_config().min_meas_per_track = 3;
        auto res_trk_cands_buffer =
            resolution_alg_cuda(traccc::get_data(trk_cands));
        track_candidate_container_types::device res_trk_cands(
            res_trk_cands_buffer);
        // All tracks are accepted as they have more than three measurements
        ASSERT_EQ(res_trk_cands.size(), 1u);

        // The first track is selected over the second one as its relative
        // shared measurement (2/7) is lower than the one of the second track
        // (2/4)
        ASSERT_EQ(get_pattern(res_trk_cands, 0),
                  std::vector<std::size_t>({1, 3, 5, 11, 14, 16, 18}));
    }
}

TEST(CudaAmbiguitySolverTests, CompareWithCPU) {}
