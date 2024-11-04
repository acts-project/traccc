/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
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

    std::array<scalar, 2> eta_range{-2.44f, 3.13f};
    const auto theta_range = eta_to_theta_range(eta_range);

    // Comparing with the values provided by
    // https://en.wikipedia.org/wiki/Pseudorapidity
    ASSERT_NEAR(theta_range[0], 5.f * detray::unit<scalar>::degree,
                0.1f * detray::unit<scalar>::degree);
    ASSERT_NEAR(theta_range[1], 170.f * detray::unit<scalar>::degree,
                0.1f * detray::unit<scalar>::degree);
}

// Test theta_to_eta_range function
TEST(ranges, theta_to_eta) {

    std::array<scalar, 2> theta_range{45.f * detray::unit<scalar>::degree,
                                      175.f * detray::unit<scalar>::degree};
    const auto eta_range = theta_to_eta_range(theta_range);

    // Comparing with the values provided by
    // https://www.star.bnl.gov/~dmitry/calc2.html
    ASSERT_NEAR(eta_range[0], -3.131f, 1e-3f);
    ASSERT_NEAR(eta_range[1], 0.881f, 1e-3f);
}
