/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "detray/utils/find_bound.hpp"

// Google Test include(s).
#include <gtest/gtest.h>

// Test upper bound function
GTEST_TEST(detray_utils, upper_bound) {

    std::vector<float> vec = {2.f, 3.f, 5.f, 8.f, 8.f, 8.f, 9.f, 12.f, 12.f};

    auto pos = detray::upper_bound(vec.begin(), vec.end(), 8.f);

    ASSERT_EQ(pos - vec.begin(), 6);
    ASSERT_EQ(*pos, 9.f);

    pos = detray::upper_bound(vec.begin(), vec.end(), 3.f);

    ASSERT_EQ(pos - vec.begin(), 2);
    ASSERT_EQ(*pos, 5.f);

    pos = detray::upper_bound(vec.begin(), vec.end(), 10.f);

    ASSERT_EQ(pos - vec.begin(), 7);
    ASSERT_EQ(*pos, 12.f);
}

// Test lower bound function
GTEST_TEST(detray_utils, lower_bound) {

    std::vector<float> vec = {2.f, 3.f, 5.f, 8.f, 8.f, 8.f, 9.f, 12.f, 12.f};

    auto pos = detray::lower_bound(vec.begin(), vec.end(), 8.f);

    ASSERT_EQ(pos - vec.begin(), 3);
    ASSERT_EQ(*pos, 8.f);

    pos = detray::lower_bound(vec.begin(), vec.end(), 3.f);

    ASSERT_EQ(pos - vec.begin(), 1);
    ASSERT_EQ(*pos, 3.f);

    pos = detray::lower_bound(vec.begin(), vec.end(), 10.f);

    ASSERT_EQ(pos - vec.begin(), 7);
    ASSERT_EQ(*pos, 12.f);
}
