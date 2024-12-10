/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/utils/unit_vectors.hpp"

// Detray test include(s)
#include "detray/test/utils/types.hpp"

// Google Test include(s)
#include <gtest/gtest.h>

using namespace detray;
using transform3 = test::transform3;
using vector3 = typename transform3::vector3;

GTEST_TEST(detray_utils, curvilinear_unit_vectors) {

    constexpr const scalar tolerance = 1e-5f;

    // General case
    vector3 w{2.f, 3.f, 4.f};

    auto uv = unit_vectors<vector3>().make_curvilinear_unit_vectors(w);
    auto u = uv[0];
    auto v = uv[1];

    EXPECT_NEAR(u[0], -3.f / getter::perp(w), tolerance);
    EXPECT_NEAR(u[1], 2.f / getter::perp(w), tolerance);
    EXPECT_NEAR(u[2], 0.f, tolerance);

    const auto test_v = vector::normalize(vector::cross(w, u));
    EXPECT_NEAR(v[0], test_v[0], tolerance);
    EXPECT_NEAR(v[1], test_v[1], tolerance);
    EXPECT_NEAR(v[2], test_v[2], tolerance);

    // Special case where w is aligned with z axis
    w = {0.f, 0.f, 23.f};

    uv = unit_vectors<vector3>().make_curvilinear_unit_vectors(w);
    u = uv[0];
    v = uv[1];

    EXPECT_NEAR(u[0], 1.f, tolerance);
    EXPECT_NEAR(u[1], 0.f, tolerance);
    EXPECT_NEAR(u[2], 0.f, tolerance);

    EXPECT_NEAR(v[0], 0.f, tolerance);
    EXPECT_NEAR(v[1], 1.f, tolerance);
    EXPECT_NEAR(v[2], 0.f, tolerance);
}
