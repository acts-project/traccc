/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/utils/ranges.hpp"

// Detray include(s).
#include "detray/definitions/units.hpp"

// GTest include(s).
#include <gtest/gtest.h>

using namespace traccc;

// Test eta_to_theta_range function
TEST(ranges, eta_to_theta) {

    std::array<scalar, 2> eta_range{-2.44, 3.13};
    const auto theta_range = eta_to_theta_range(eta_range);

    // Comparing with the values provided by
    // https://en.wikipedia.org/wiki/Pseudorapidity
    ASSERT_NEAR(theta_range[0], 5.f * detray::unit<scalar>::degree,
                0.1f * detray::unit<scalar>::degree);
    ASSERT_NEAR(theta_range[1], 170.f * detray::unit<scalar>::degree,
                0.1f * detray::unit<scalar>::degree);
}