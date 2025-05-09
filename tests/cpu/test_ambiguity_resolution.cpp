/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/ambiguity_resolution/legacy/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

#include <chrono>
#include <random>

using namespace traccc;

namespace {
vecmem::host_memory_resource host_mr;
traccc::memory_resource mr{host_mr, &host_mr};

}  // namespace

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
    fill_pattern(trk_cands, 0, 0.23f, {1, 3, 5, 11});
    fill_pattern(trk_cands, 1, 0.85f, {6, 7, 8, 9, 10, 12});
    fill_pattern(trk_cands, 2, 0.42f, {2, 4, 13});

    traccc::host::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;

    traccc::host::greedy_ambiguity_resolution_algorithm resolution_alg(
        resolution_config, mr);
    {
        resolution_alg.get_config().min_meas_per_track = 3;
        auto res_trk_cands = resolution_alg(traccc::get_data(trk_cands));
        // All tracks are accepted as they have more than three measurements
        ASSERT_EQ(res_trk_cands.size(), 3u);
        ASSERT_EQ(get_pattern(res_trk_cands, 0),
                  std::vector<std::size_t>({1, 3, 5, 11}));
        ASSERT_EQ(get_pattern(res_trk_cands, 1),
                  std::vector<std::size_t>({6, 7, 8, 9, 10, 12}));
        ASSERT_EQ(get_pattern(res_trk_cands, 2),
                  std::vector<std::size_t>({2, 4, 13}));
    }

    {
        resolution_alg.get_config().min_meas_per_track = 5;
        auto res_trk_cands = resolution_alg(traccc::get_data(trk_cands));
        // Only the second track with six measurements is accepted
        ASSERT_EQ(res_trk_cands.size(), 1u);
        ASSERT_EQ(get_pattern(res_trk_cands, 0),
                  std::vector<std::size_t>({6, 7, 8, 9, 10, 12}));
    }

    /*******************
     * Legacy algorithm
     * *****************/

    {
        traccc::legacy::greedy_ambiguity_resolution_algorithm::config_t
            legacy_cfg;
        traccc::legacy::greedy_ambiguity_resolution_algorithm
            legacy_resolution_alg(legacy_cfg);

        legacy_resolution_alg.get_config().n_measurements_min = 3;
        auto res_trk_cands = legacy_resolution_alg(trk_cands);
        ASSERT_EQ(res_trk_cands.size(), 3u);
    }

    {
        traccc::legacy::greedy_ambiguity_resolution_algorithm::config_t
            legacy_cfg;
        traccc::legacy::greedy_ambiguity_resolution_algorithm
            legacy_resolution_alg(legacy_cfg);

        legacy_resolution_alg.get_config().n_measurements_min = 5;
        auto res_trk_cands = legacy_resolution_alg(trk_cands);
        ASSERT_EQ(res_trk_cands.size(), 1u);
        ASSERT_EQ(get_pattern(res_trk_cands, 0),
                  std::vector<std::size_t>({6, 7, 8, 9, 10, 12}));
    }
}

TEST(AmbiguitySolverTests, GreedyResolverTest1) {

    track_candidate_container_types::host trk_cands;

    trk_cands.resize(2u);
    fill_pattern(trk_cands, 0, 0.12f, {1, 3, 5, 11, 14, 16, 18});
    fill_pattern(trk_cands, 1, 0.53f, {3, 5, 6, 13});

    traccc::host::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::host::greedy_ambiguity_resolution_algorithm resolution_alg(
        resolution_config, mr);
    {
        auto res_trk_cands = resolution_alg(traccc::get_data(trk_cands));
        ASSERT_EQ(res_trk_cands.size(), 1u);

        // The first track is selected over the second one as its relative
        // shared measurement (2/7) is lower than the one of the second track
        // (2/4)
        ASSERT_EQ(get_pattern(res_trk_cands, 0),
                  std::vector<std::size_t>({1, 3, 5, 11, 14, 16, 18}));
    }

    /*******************
     * Legacy algorithm
     * *****************/

    {
        traccc::legacy::greedy_ambiguity_resolution_algorithm::config_t
            legacy_cfg;
        traccc::legacy::greedy_ambiguity_resolution_algorithm
            legacy_resolution_alg(legacy_cfg);

        auto res_trk_cands = legacy_resolution_alg(trk_cands);
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
    fill_pattern(trk_cands, 0, 0.8f, {1, 3, 5, 11});
    fill_pattern(trk_cands, 1, 0.9f, {3, 5, 6, 13});

    traccc::host::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::host::greedy_ambiguity_resolution_algorithm resolution_alg(
        resolution_config, mr);
    {
        auto res_trk_cands = resolution_alg(traccc::get_data(trk_cands));
        ASSERT_EQ(res_trk_cands.size(), 1u);

        // The second track is selected over the first one as their relative
        // shared measurement (2/4) is the same but its p-value is higher
        ASSERT_EQ(get_pattern(res_trk_cands, 0),
                  std::vector<std::size_t>({3, 5, 6, 13}));
    }
}

TEST(AmbiguitySolverTests, GreedyResolverTest3) {

    track_candidate_container_types::host trk_cands;
    trk_cands.resize(6u);
    fill_pattern(trk_cands, 0, 0.2f, {1, 3, 5, 11});
    fill_pattern(trk_cands, 1, 0.5f, {2, 6});
    fill_pattern(trk_cands, 2, 0.4f, {3, 6, 12, 14, 19, 21});
    fill_pattern(trk_cands, 3, 0.1f, {2, 7, 11, 13, 16});
    fill_pattern(trk_cands, 4, 0.3f, {1, 7, 8});
    fill_pattern(trk_cands, 5, 0.6f, {1, 3, 11, 22});

    traccc::host::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::host::greedy_ambiguity_resolution_algorithm resolution_alg(
        resolution_config, mr);

    {
        auto res_trk_cands = resolution_alg(traccc::get_data(trk_cands));
        ASSERT_EQ(res_trk_cands.size(), 2u);

        ASSERT_EQ(get_pattern(res_trk_cands, 0),
                  std::vector<std::size_t>({3, 6, 12, 14, 19, 21}));
        ASSERT_EQ(get_pattern(res_trk_cands, 1),
                  std::vector<std::size_t>({2, 7, 11, 13, 16}));
    }

    // Legacy algorithm
    traccc::legacy::greedy_ambiguity_resolution_algorithm::config_t legacy_cfg;
    traccc::legacy::greedy_ambiguity_resolution_algorithm legacy_resolution_alg(
        legacy_cfg);

    {
        auto res_trk_cands = legacy_resolution_alg(trk_cands);
        ASSERT_EQ(res_trk_cands.size(), 2u);

        ASSERT_EQ(get_pattern(res_trk_cands, 0),
                  std::vector<std::size_t>({3, 6, 12, 14, 19, 21}));
        ASSERT_EQ(get_pattern(res_trk_cands, 1),
                  std::vector<std::size_t>({2, 7, 11, 13, 16}));
    }
}

// Comparison to the legacy algorithm.
TEST(AmbiguitySolverTests, GreedyResolverTest4) {

    std::size_t n_tracks = 10000u;

    track_candidate_container_types::host trk_cands;
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

    traccc::host::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::host::greedy_ambiguity_resolution_algorithm resolution_alg(
        resolution_config, mr);

    auto start_new = std::chrono::high_resolution_clock::now();

    auto res_trk_cands = resolution_alg(traccc::get_data(trk_cands));

    auto end_new = std::chrono::high_resolution_clock::now();
    auto duration_new = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_new - start_new);
    std::cout << " Time for the new method " << duration_new.count() << " ms"
              << std::endl;

    // Legacy algorithm
    traccc::legacy::greedy_ambiguity_resolution_algorithm::config_t legacy_cfg;
    traccc::legacy::greedy_ambiguity_resolution_algorithm legacy_resolution_alg(
        legacy_cfg);

    auto start_legacy = std::chrono::high_resolution_clock::now();

    auto legacy_res_trk_cands = legacy_resolution_alg(trk_cands);

    auto end_legacy = std::chrono::high_resolution_clock::now();
    auto duration_legacy =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_legacy -
                                                              start_legacy);
    std::cout << " Time for the legacy method " << duration_legacy.count()
              << " ms" << std::endl;

    std::size_t n_res_tracks = res_trk_cands.size();
    ASSERT_EQ(n_res_tracks, legacy_res_trk_cands.size());
    for (std::size_t i = 0; i < n_res_tracks; i++) {
        ASSERT_EQ(res_trk_cands.at(i).header,
                  legacy_res_trk_cands.at(i).header);
        ASSERT_EQ(res_trk_cands.at(i).items, legacy_res_trk_cands.at(i).items);
    }
}
