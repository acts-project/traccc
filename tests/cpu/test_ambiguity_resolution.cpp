/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/ambiguity_resolution/legacy/greedy_ambiguity_resolution_algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

#include <random>

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

    std::random_device rd;
    std::mt19937 seed_gen(rd());
    std::uniform_int_distribution<> seed_dist(0, 1000);

    // int seed = seed_dist(seed_gen);
    // std::cout << "Seed: " << seed << std::endl;

    std::size_t n_tracks = 5u;

    track_candidate_container_types::host trk_cands;
    trk_cands.resize(n_tracks);
    // std::mt19937 gen(seed);
    std::mt19937 gen(953);

    for (std::size_t i = 0; i < n_tracks; i++) {

        std::uniform_int_distribution<std::size_t> track_length_dist(3, 5);
        std::uniform_int_distribution<std::size_t> meas_id_dist(0, 10);
        std::uniform_real_distribution<traccc::scalar> chi2_dist(0.0f, 10.0f);

        const std::size_t track_length = track_length_dist(gen);
        const traccc::scalar chi2 = chi2_dist(gen);
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
        fill_pattern(trk_cands, i, chi2, pattern);
    }

    traccc::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::greedy_ambiguity_resolution_algorithm resolution_alg(
        resolution_config);
    auto res_trk_cands = resolution_alg(trk_cands);

    // Legacy algorithm
    traccc::legacy::greedy_ambiguity_resolution_algorithm::config_t legacy_cfg;
    traccc::legacy::greedy_ambiguity_resolution_algorithm legacy_resolution_alg(
        legacy_cfg);
    auto legacy_res_trk_cands = legacy_resolution_alg(trk_cands);

    /*
    ASSERT_EQ(res_trk_cands.size(), legacy_res_trk_cands.size());
    for (std::size_t i = 0; i < res_trk_cands.size(); i++) {
        ASSERT_EQ(res_trk_cands.at(i).items, legacy_res_trk_cands.at(i).items);
    }
    */

    std::cout << res_trk_cands.size() << " " << legacy_res_trk_cands.size()
              << std::endl;
    EXPECT_EQ(res_trk_cands.size(), legacy_res_trk_cands.size());

    std::cout << "Event " << std::endl;
    for (std::size_t i = 0; i < trk_cands.size(); i++) {
        auto chi2 = trk_cands.at(i).header.trk_quality.chi2;
        auto pattern = get_pattern(trk_cands, i);

        std::cout << "Chi2: " << chi2 << " | ";
        for (std::size_t j = 0; j < pattern.size(); j++) {
            std::cout << pattern.at(j) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "New algorithm " << std::endl << std::endl;

    for (std::size_t i = 0; i < res_trk_cands.size(); i++) {
        auto chi2 = res_trk_cands.at(i).header.trk_quality.chi2;
        auto pattern = get_pattern(res_trk_cands, i);

        std::cout << "Chi2: " << chi2 << " | ";
        for (std::size_t j = 0; j < pattern.size(); j++) {
            std::cout << pattern.at(j) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "legacy algorithm " << std::endl << std::endl;

    for (std::size_t i = 0; i < legacy_res_trk_cands.size(); i++) {
        auto chi2 = legacy_res_trk_cands.at(i).header.trk_quality.chi2;
        auto pattern = get_pattern(legacy_res_trk_cands, i);

        std::cout << "Chi2: " << chi2 << " | ";
        for (std::size_t j = 0; j < pattern.size(); j++) {
            std::cout << pattern.at(j) << " ";
        }
        std::cout << std::endl;
    }
}
