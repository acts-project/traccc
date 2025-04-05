/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

using namespace traccc;

namespace {
vecmem::host_memory_resource host_mr;
}

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

TEST(AmbiguitySolverTests, GreedyResolverTest0) {

    track_candidate_container_types::host trk_cands;

    trk_cands.resize(3u);
    fill_pattern(trk_cands, 0, 8.3f, {1, 3, 5, 11});
    fill_pattern(trk_cands, 1, 2.2f, {6, 7, 8, 9, 10, 12});
    fill_pattern(trk_cands, 2, 3.7f, {2, 4, 13});

    traccc::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;

    traccc::greedy_ambiguity_resolution_algorithm resolution_alg(
        resolution_config);
    {
        resolution_alg.get_config().min_meas_per_track = 3;
        auto res_trk_cands = resolution_alg(trk_cands);
        // All tracks are accepted as they have more than three measurements
        ASSERT_EQ(res_trk_cands.size(), 3u);
    }

    {
        resolution_alg.get_config().min_meas_per_track = 5;
        auto res_trk_cands = resolution_alg(trk_cands);
        // Only the second track with six measurements is accepted
        ASSERT_EQ(res_trk_cands.size(), 1u);
        ASSERT_EQ(get_pattern(res_trk_cands, 0),
                  std::vector<std::size_t>({6, 7, 8, 9, 10, 12}));
    }
}

TEST(AmbiguitySolverTests, GreedyResolverTest1) {

    track_candidate_container_types::host trk_cands;

    trk_cands.resize(2u);
    fill_pattern(trk_cands, 0, 4.3f, {1, 3, 5, 11, 14, 16, 18});
    fill_pattern(trk_cands, 1, 1.2f, {3, 5, 6, 13});

    traccc::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::greedy_ambiguity_resolution_algorithm resolution_alg(
        resolution_config);
    {
        auto res_trk_cands = resolution_alg(trk_cands);
        ASSERT_EQ(res_trk_cands.size(), 1u);

        // The first track is selected over the second one as its relative
        // shared measurement (2/7) is lower than the one of the second track
        // (2/4)
        ASSERT_EQ(get_pattern(res_trk_cands, 0),
                  std::vector<std::size_t>({1, 3, 5, 11, 14, 16, 18}));
    }
}

TEST(AmbiguitySolverTests, GreedyResolverTest2) {

    track_candidate_container_types::host trk_cands;

    trk_cands.resize(2u);
    fill_pattern(trk_cands, 0, 4.3f, {1, 3, 5, 11});
    fill_pattern(trk_cands, 1, 1.2f, {3, 5, 6, 13});

    traccc::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::greedy_ambiguity_resolution_algorithm resolution_alg(
        resolution_config);
    {
        auto res_trk_cands = resolution_alg(trk_cands);
        ASSERT_EQ(res_trk_cands.size(), 1u);

        // The second track is selected over the first one as their relative
        // shared measurement (2/4) is the same but its chi square is smaller
        ASSERT_EQ(get_pattern(res_trk_cands, 0),
                  std::vector<std::size_t>({3, 5, 6, 13}));
    }
}

TEST(AmbiguitySolverTests, GreedyResolverTest3) {

    track_candidate_container_types::host trk_cands;
    trk_cands.resize(6u);
    fill_pattern(trk_cands, 0, 5.3f, {1, 3, 5, 11});
    fill_pattern(trk_cands, 1, 2.4f, {2, 6});
    fill_pattern(trk_cands, 2, 2.5f, {3, 6, 12, 14, 19, 21});
    fill_pattern(trk_cands, 3, 13.3f, {2, 7, 11, 13, 16});
    fill_pattern(trk_cands, 4, 4.1f, {1, 7, 8});
    fill_pattern(trk_cands, 5, 1.1f, {1, 3, 11, 22});

    // Relative shared measurement number
    // 0-th track: 3/4 (sharing with 2, 3, 5) => Reject with high shared meas
    // 1-th track: 2/2 (sharing with 2, 3) => Reject with only two measurements
    // 2-th track: 2/6 (sharing with 0, 1, 5) => Pass
    // 3-th track: 3/5 (sharing with 0, 1, 4, 5) => Pass
    // 4-th track: 2/3 (sharing with 0, 3) => Reject with high shared meas
    // 5-th track: 3/4 (sharing with 0, 2) => Reject with high shared meas

    traccc::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::greedy_ambiguity_resolution_algorithm resolution_alg(
        resolution_config);

    {
        auto res_trk_cands = resolution_alg(trk_cands);
        ASSERT_EQ(res_trk_cands.size(), 2u);

        ASSERT_EQ(get_pattern(res_trk_cands, 0),
                  std::vector<std::size_t>({3, 6, 12, 14, 19, 21}));
        ASSERT_EQ(get_pattern(res_trk_cands, 1),
                  std::vector<std::size_t>({2, 7, 11, 13, 16}));
    }
}
