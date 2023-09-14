/**
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/options.hpp"

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <sstream>

TEST(options, parsing) {

    traccc::Reals<int, 3> parsed_vars;

    std::istringstream I("100:200:300");
    I >> parsed_vars;

    ASSERT_EQ(parsed_vars[0], 100);
    ASSERT_EQ(parsed_vars[1], 200);
    ASSERT_EQ(parsed_vars[2], 300);
}
