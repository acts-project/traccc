/**
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/utils/compare.hpp"

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <algorithm>
#include <utility>
#include <vector>

TEST(compare, binary_search) {

    std::vector<std::pair<int, int> > vec;
    vec.push_back({1, 2});
    vec.push_back({3, 4});
    vec.push_back({3, 7});
    vec.push_back({5, 1});
    vec.push_back({9, 3});

    auto rg = std::equal_range(vec.begin(), vec.end(), 3,
                               traccc::compare_pair_int<std::pair, int>());
    ASSERT_EQ((*rg.first).second, 4);
    ASSERT_EQ((*rg.second).second, 1);

    rg = std::equal_range(vec.begin(), vec.end(), 5,
                          traccc::compare_pair_int<std::pair, int>());
    ASSERT_EQ((*rg.first).second, 1);
    ASSERT_EQ((*rg.second).second, 3);

    auto lower = std::lower_bound(vec.begin(), vec.end(), 3,
                                  traccc::compare_pair_int<std::pair, int>());
    ASSERT_EQ((*lower).second, 4);

    lower = std::lower_bound(vec.begin(), vec.end(), 2,
                             traccc::compare_pair_int<std::pair, int>());
    ASSERT_EQ(std::distance(vec.begin(), lower), 1);
    ASSERT_EQ((*lower).second, 4);

    auto upper = std::upper_bound(vec.begin(), vec.end(), 3,
                                  traccc::compare_pair_int<std::pair, int>());
    ASSERT_EQ((*upper).second, 1);

    upper = std::upper_bound(vec.begin(), vec.end(), 2,
                             traccc::compare_pair_int<std::pair, int>());
    ASSERT_EQ((*upper).second, 4);
}
