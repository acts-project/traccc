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
}  // namespace

void fill_pattern(
    edm::track_candidate_collection<default_algebra>::host& track_candidates,
    measurement_collection_types::host& measurements, const std::size_t idx,
    const traccc::scalar pval, const std::vector<std::size_t>& pattern) {

    track_candidates.at(idx).pval() = pval;

    for (const auto& meas_id : pattern) {
        measurements.emplace_back();
        measurements.back().measurement_id = meas_id;
        track_candidates.at(idx).measurement_indices().push_back(
            static_cast<unsigned int>(measurements.size() - 1));
    }
}

std::vector<std::size_t> get_pattern(
    const edm::track_candidate_collection<default_algebra>::host&
        track_candidates,
    const measurement_collection_types::host& measurements,
    const std::size_t idx) {
    std::vector<std::size_t> ret;
    // A const reference would be fine here. But GCC fears that that would lead
    // to a dangling reference...
    const auto meas_indices = track_candidates.at(idx).measurement_indices();
    for (unsigned int meas_idx : meas_indices) {
        ret.push_back(measurements.at(meas_idx).measurement_id);
    }

    return ret;
}

TEST(AmbiguitySolverTests, GreedyResolverTest0) {

    measurement_collection_types::host measurements{&host_mr};
    edm::track_candidate_collection<default_algebra>::host trk_cands{host_mr};

    trk_cands.resize(3u);
    fill_pattern(trk_cands, measurements, 0, 0.23f, {5, 1, 11, 3});
    fill_pattern(trk_cands, measurements, 1, 0.85f, {12, 10, 9, 8, 7, 6});
    fill_pattern(trk_cands, measurements, 2, 0.42f, {4, 2, 13});

    traccc::host::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;

    traccc::host::greedy_ambiguity_resolution_algorithm resolution_alg(
        resolution_config, host_mr);
    {
        resolution_alg.get_config().min_meas_per_track = 3;
        auto res_trk_cands = resolution_alg(vecmem::get_data(trk_cands),
                                            vecmem::get_data(measurements));
        // All tracks are accepted as they have more than three measurements
        ASSERT_EQ(res_trk_cands.size(), 3u);
        ASSERT_EQ(get_pattern(res_trk_cands, measurements, 0),
                  std::vector<std::size_t>({5, 1, 11, 3}));
        ASSERT_EQ(get_pattern(res_trk_cands, measurements, 1),
                  std::vector<std::size_t>({12, 10, 9, 8, 7, 6}));
        ASSERT_EQ(get_pattern(res_trk_cands, measurements, 2),
                  std::vector<std::size_t>({4, 2, 13}));
    }

    {
        resolution_alg.get_config().min_meas_per_track = 5;
        auto res_trk_cands = resolution_alg(vecmem::get_data(trk_cands),
                                            vecmem::get_data(measurements));
        // Only the second track with six measurements is accepted
        ASSERT_EQ(res_trk_cands.size(), 1u);
        ASSERT_EQ(get_pattern(res_trk_cands, measurements, 0),
                  std::vector<std::size_t>({12, 10, 9, 8, 7, 6}));
    }

    /*******************
     * Legacy algorithm
     * *****************/

    {
        traccc::legacy::greedy_ambiguity_resolution_algorithm::config_t
            legacy_cfg;
        traccc::legacy::greedy_ambiguity_resolution_algorithm
            legacy_resolution_alg(legacy_cfg, host_mr);

        legacy_resolution_alg.get_config().n_measurements_min = 3;
        auto res_trk_cands = legacy_resolution_alg(trk_cands, measurements);
        ASSERT_EQ(res_trk_cands.size(), 3u);
        ASSERT_EQ(get_pattern(res_trk_cands, measurements, 0),
                  std::vector<std::size_t>({5, 1, 11, 3}));
        ASSERT_EQ(get_pattern(res_trk_cands, measurements, 1),
                  std::vector<std::size_t>({12, 10, 9, 8, 7, 6}));
        ASSERT_EQ(get_pattern(res_trk_cands, measurements, 2),
                  std::vector<std::size_t>({4, 2, 13}));
    }

    {
        traccc::legacy::greedy_ambiguity_resolution_algorithm::config_t
            legacy_cfg;
        traccc::legacy::greedy_ambiguity_resolution_algorithm
            legacy_resolution_alg(legacy_cfg, host_mr);

        legacy_resolution_alg.get_config().n_measurements_min = 5;
        auto res_trk_cands = legacy_resolution_alg(trk_cands, measurements);
        ASSERT_EQ(res_trk_cands.size(), 1u);
        ASSERT_EQ(get_pattern(res_trk_cands, measurements, 0),
                  std::vector<std::size_t>({12, 10, 9, 8, 7, 6}));
    }
}

TEST(AmbiguitySolverTests, GreedyResolverTest1) {

    measurement_collection_types::host measurements{&host_mr};
    edm::track_candidate_collection<default_algebra>::host trk_cands{host_mr};

    trk_cands.resize(2u);
    fill_pattern(trk_cands, measurements, 0, 0.12f, {5, 14, 1, 11, 18, 16, 3});
    fill_pattern(trk_cands, measurements, 1, 0.53f, {3, 6, 5, 13});

    traccc::host::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::host::greedy_ambiguity_resolution_algorithm resolution_alg(
        resolution_config, host_mr);
    {
        auto res_trk_cands = resolution_alg(vecmem::get_data(trk_cands),
                                            vecmem::get_data(measurements));
        ASSERT_EQ(res_trk_cands.size(), 1u);

        // The first track is selected over the second one as its relative
        // shared measurement (2/7) is lower than the one of the second track
        // (2/4)
        ASSERT_EQ(get_pattern(res_trk_cands, measurements, 0),
                  std::vector<std::size_t>({5, 14, 1, 11, 18, 16, 3}));
    }

    /*******************
     * Legacy algorithm
     * *****************/

    {
        traccc::legacy::greedy_ambiguity_resolution_algorithm::config_t
            legacy_cfg;
        traccc::legacy::greedy_ambiguity_resolution_algorithm
            legacy_resolution_alg(legacy_cfg, host_mr);

        auto res_trk_cands = legacy_resolution_alg(trk_cands, measurements);
        ASSERT_EQ(res_trk_cands.size(), 1u);

        // The first track is selected over the second one as its relative
        // shared measurement (2/7) is lower than the one of the second track
        // (2/4)
        ASSERT_EQ(get_pattern(res_trk_cands, measurements, 0),
                  std::vector<std::size_t>({5, 14, 1, 11, 18, 16, 3}));
    }
}

TEST(AmbiguitySolverTests, GreedyResolverTest2) {

    measurement_collection_types::host measurements{&host_mr};
    edm::track_candidate_collection<default_algebra>::host trk_cands{host_mr};

    trk_cands.resize(2u);
    fill_pattern(trk_cands, measurements, 0, 0.8f, {1, 3, 5, 11});
    fill_pattern(trk_cands, measurements, 1, 0.9f, {3, 5, 6, 13});

    traccc::host::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::host::greedy_ambiguity_resolution_algorithm resolution_alg(
        resolution_config, host_mr);
    {
        auto res_trk_cands = resolution_alg(vecmem::get_data(trk_cands),
                                            vecmem::get_data(measurements));
        ASSERT_EQ(res_trk_cands.size(), 1u);

        // The second track is selected over the first one as their relative
        // shared measurement (2/4) is the same but its p-value is higher
        ASSERT_EQ(get_pattern(res_trk_cands, measurements, 0),
                  std::vector<std::size_t>({3, 5, 6, 13}));
    }
}

TEST(AmbiguitySolverTests, GreedyResolverTest3) {

    measurement_collection_types::host measurements{&host_mr};
    edm::track_candidate_collection<default_algebra>::host trk_cands{host_mr};

    trk_cands.resize(6u);
    fill_pattern(trk_cands, measurements, 0, 0.2f, {5, 1, 11, 3});
    fill_pattern(trk_cands, measurements, 1, 0.5f, {6, 2});
    fill_pattern(trk_cands, measurements, 2, 0.4f, {3, 21, 12, 6, 19, 14});
    fill_pattern(trk_cands, measurements, 3, 0.1f, {13, 16, 2, 7, 11});
    fill_pattern(trk_cands, measurements, 4, 0.3f, {1, 7, 8});
    fill_pattern(trk_cands, measurements, 5, 0.6f, {1, 3, 11, 22});

    traccc::host::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::host::greedy_ambiguity_resolution_algorithm resolution_alg(
        resolution_config, host_mr);

    {
        auto res_trk_cands = resolution_alg(vecmem::get_data(trk_cands),
                                            vecmem::get_data(measurements));
        ASSERT_EQ(res_trk_cands.size(), 2u);

        ASSERT_EQ(get_pattern(res_trk_cands, measurements, 0),
                  std::vector<std::size_t>({3, 21, 12, 6, 19, 14}));
        ASSERT_EQ(get_pattern(res_trk_cands, measurements, 1),
                  std::vector<std::size_t>({13, 16, 2, 7, 11}));
    }

    // Legacy algorithm
    traccc::legacy::greedy_ambiguity_resolution_algorithm::config_t legacy_cfg;
    traccc::legacy::greedy_ambiguity_resolution_algorithm legacy_resolution_alg(
        legacy_cfg, host_mr);

    {
        auto res_trk_cands = legacy_resolution_alg(trk_cands, measurements);
        ASSERT_EQ(res_trk_cands.size(), 2u);

        ASSERT_EQ(get_pattern(res_trk_cands, measurements, 0),
                  std::vector<std::size_t>({3, 21, 12, 6, 19, 14}));
        ASSERT_EQ(get_pattern(res_trk_cands, measurements, 1),
                  std::vector<std::size_t>({13, 16, 2, 7, 11}));
    }
}

// Comparison to the legacy algorithm.
TEST(AmbiguitySolverTests, GreedyResolverTest4) {

    std::size_t n_tracks = 10000u;

    measurement_collection_types::host measurements{&host_mr};
    edm::track_candidate_collection<default_algebra>::host trk_cands{host_mr};
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
        fill_pattern(trk_cands, measurements, i, pval, pattern);
    }

    traccc::host::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::host::greedy_ambiguity_resolution_algorithm resolution_alg(
        resolution_config, host_mr);

    auto start_new = std::chrono::high_resolution_clock::now();

    auto res_trk_cands = resolution_alg(vecmem::get_data(trk_cands),
                                        vecmem::get_data(measurements));

    auto end_new = std::chrono::high_resolution_clock::now();
    auto duration_new = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_new - start_new);
    std::cout << " Time for the new method " << duration_new.count() << " ms"
              << std::endl;

    // Legacy algorithm
    traccc::legacy::greedy_ambiguity_resolution_algorithm::config_t legacy_cfg;
    traccc::legacy::greedy_ambiguity_resolution_algorithm legacy_resolution_alg(
        legacy_cfg, host_mr);

    auto start_legacy = std::chrono::high_resolution_clock::now();

    auto legacy_res_trk_cands = legacy_resolution_alg(trk_cands, measurements);

    auto end_legacy = std::chrono::high_resolution_clock::now();
    auto duration_legacy =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_legacy -
                                                              start_legacy);
    std::cout << " Time for the legacy method " << duration_legacy.count()
              << " ms" << std::endl;

    std::size_t n_res_tracks = res_trk_cands.size();
    ASSERT_EQ(n_res_tracks, legacy_res_trk_cands.size());
    for (std::size_t i = 0; i < n_res_tracks; i++) {
        ASSERT_EQ(res_trk_cands.at(i), legacy_res_trk_cands.at(i));
    }
}

TEST(AmbiguitySolverTests, GreedyResolverTest5) {

    measurement_collection_types::host measurements{&host_mr};
    edm::track_candidate_collection<default_algebra>::host trk_cands{host_mr};
    trk_cands.resize(4u);
    fill_pattern(trk_cands, measurements, 0, 0.2f, {1, 2, 1, 1});
    fill_pattern(trk_cands, measurements, 1, 0.5f, {3, 2, 1});
    fill_pattern(trk_cands, measurements, 2, 0.4f, {2, 4, 5, 7, 2});
    fill_pattern(trk_cands, measurements, 3, 0.1f, {6, 6, 6, 6});

    traccc::host::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::host::greedy_ambiguity_resolution_algorithm resolution_alg(
        resolution_config, host_mr);

    {
        auto res_trk_cands = resolution_alg(vecmem::get_data(trk_cands),
                                            vecmem::get_data(measurements));
        ASSERT_EQ(res_trk_cands.size(), 2u);

        ASSERT_EQ(get_pattern(res_trk_cands, measurements, 0),
                  std::vector<std::size_t>({3, 2, 1}));
        ASSERT_EQ(get_pattern(res_trk_cands, measurements, 1),
                  std::vector<std::size_t>({6, 6, 6, 6}));
    }
}

TEST(AmbiguitySolverTests, GreedyResolverTest6) {

    measurement_collection_types::host measurements{&host_mr};
    edm::track_candidate_collection<default_algebra>::host trk_cands{host_mr};
    trk_cands.resize(5u);
    fill_pattern(trk_cands, measurements, 0, 0.2f, {7, 3, 5, 7, 7, 7, 2});
    fill_pattern(trk_cands, measurements, 1, 0.5f, {2});
    fill_pattern(trk_cands, measurements, 2, 0.4f, {8, 9, 7, 2, 3, 4, 3, 7});
    fill_pattern(trk_cands, measurements, 3, 0.1f, {8, 9, 0, 8, 1, 4, 6});
    fill_pattern(trk_cands, measurements, 4, 0.9f, {10, 3, 2});

    traccc::host::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config;
    traccc::host::greedy_ambiguity_resolution_algorithm resolution_alg(
        resolution_config, host_mr);

    {
        auto res_trk_cands = resolution_alg(vecmem::get_data(trk_cands),
                                            vecmem::get_data(measurements));
        ASSERT_EQ(res_trk_cands.size(), 2u);

        ASSERT_EQ(get_pattern(res_trk_cands, measurements, 0),
                  std::vector<std::size_t>({7, 3, 5, 7, 7, 7, 2}));
        ASSERT_EQ(get_pattern(res_trk_cands, measurements, 1),
                  std::vector<std::size_t>({8, 9, 0, 8, 1, 4, 6}));
    }
}
