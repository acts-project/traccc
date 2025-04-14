/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/unique_ptr.hpp>

using namespace traccc;

namespace {

// Memory resource used by the EDM.
vecmem::cuda::managed_memory_resource mng_mr;
traccc::memory_resource mr{mng_mr};

}  // namespace


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

std::vector<std::size_t> get_pattern(
    track_candidate_container_types::host& track_candidates,
    const std::size_t idx) {
    std::vector<std::size_t> ret;
    auto& cands = track_candidates.at(idx).items;
    for (const auto& cand : cands) {
        ret.push_back(cand.measurement_id);
    }

    return ret;
}

TEST(CudaAmbiguitySolverTests, GreedyResolverTest0) {

    track_candidate_container_types::host trk_cands;

    trk_cands.resize(3u);
    fill_pattern(trk_cands, 0, 8.3f, {1, 3, 5, 11});
    fill_pattern(trk_cands, 1, 2.2f, {6, 7, 8, 9, 10, 12});
    fill_pattern(trk_cands, 2, 3.7f, {2, 4, 13});

    traccc::host::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;

    traccc::host::greedy_ambiguity_resolution_algorithm resolution_alg(
        resolution_config, mr);
    {
        resolution_alg.get_config().min_meas_per_track = 3;
        auto res_trk_cands = resolution_alg(traccc::get_data(trk_cands));
        // All tracks are accepted as they have more than three measurements
        ASSERT_EQ(res_trk_cands.size(), 3u);
    }

    {
        resolution_alg.get_config().min_meas_per_track = 5;
        auto res_trk_cands = resolution_alg(traccc::get_data(trk_cands));
        // Only the second track with six measurements is accepted
        ASSERT_EQ(res_trk_cands.size(), 1u);
        ASSERT_EQ(get_pattern(res_trk_cands, 0),
                  std::vector<std::size_t>({6, 7, 8, 9, 10, 12}));
    }
}

TEST(CudaAmbiguitySolverTests, CompareWithCPU) {


}