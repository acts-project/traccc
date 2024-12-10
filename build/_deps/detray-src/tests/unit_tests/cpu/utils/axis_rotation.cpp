/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "detray/utils/axis_rotation.hpp"

#include "detray/definitions/units.hpp"

// Detray test include(s)
#include "detray/test/utils/types.hpp"

// GTest include(s).
#include <gtest/gtest.h>

using namespace detray;

using algebra_t = test::algebra;
using scalar_t = test::scalar;
using matrix_operator = test::matrix_operator;
template <std::size_t ROWS, std::size_t COLS>
using matrix_type = test::matrix<ROWS, COLS>;

static constexpr scalar isclose{1e-5f};

GTEST_TEST(detray_utils, axis_rotation) {

    const test::vector3 axis{0.f, 0.f, 3.f};

    const test::vector3 v1{1.f, 0.f, 0.f};

    const auto u1 =
        axis_rotation<algebra_t>(axis, constant<scalar_t>::pi_2)(v1);

    EXPECT_NEAR(u1[0], 0.f, isclose);
    EXPECT_NEAR(u1[1], 1.f, isclose);
    EXPECT_NEAR(u1[2], 0.f, isclose);

    matrix_type<3, 1> v2;
    matrix_operator().element(v2, 0, 0) = constant<scalar_t>::inv_sqrt2;
    matrix_operator().element(v2, 1, 0) = constant<scalar_t>::inv_sqrt2;
    matrix_operator().element(v2, 2, 0) = 0.f;

    const auto u2 =
        axis_rotation<algebra_t>(axis, constant<scalar_t>::pi_4)(v2);

    EXPECT_NEAR(matrix_operator().element(u2, 0, 0), 0.f, isclose);
    EXPECT_NEAR(matrix_operator().element(u2, 1, 0), 1.f, isclose);
    EXPECT_NEAR(matrix_operator().element(u2, 2, 0), 0.f, isclose);

    // Counter clockswise pi/4-Rotation of (1,0,0) around (0,0,1) ->
    // (inv_sqrt2,inv_sqrt2,0)
    const test::vector3 v3{1.f, 0.f, 0.f};

    const auto u3 =
        axis_rotation<algebra_t>(axis, constant<scalar_t>::pi_4)(v3);

    EXPECT_NEAR(u3[0u], constant<scalar_t>::inv_sqrt2, isclose);
    EXPECT_NEAR(u3[1u], constant<scalar_t>::inv_sqrt2, isclose);
    EXPECT_NEAR(u3[2u], 0.f, isclose);
}

GTEST_TEST(detray_utils, euler_rotation1) {

    euler_rotation<algebra_t> euler_rot;
    euler_rot.alpha = constant<scalar_t>::pi_2;

    auto [x1, z1] = euler_rot();

    EXPECT_NEAR(x1[0], 0.f, isclose);
    EXPECT_NEAR(x1[1], 1.f, isclose);
    EXPECT_NEAR(x1[2], 0.f, isclose);

    EXPECT_NEAR(z1[0], 0.f, isclose);
    EXPECT_NEAR(z1[1], 0.f, isclose);
    EXPECT_NEAR(z1[2], 1.f, isclose);

    euler_rot.beta = constant<scalar_t>::pi_2;

    auto [x2, z2] = euler_rot();
    EXPECT_NEAR(x2[0], 0.f, isclose);
    EXPECT_NEAR(x2[1], 1.f, isclose);
    EXPECT_NEAR(x2[2], 0.f, isclose);

    EXPECT_NEAR(z2[0], 1.f, isclose);
    EXPECT_NEAR(z2[1], 0.f, isclose);
    EXPECT_NEAR(z2[2], 0.f, isclose);

    euler_rot.gamma = constant<scalar_t>::pi_2;

    auto [x3, z3] = euler_rot();
    EXPECT_NEAR(x3[0], 0.f, isclose);
    EXPECT_NEAR(x3[1], 0.f, isclose);
    EXPECT_NEAR(x3[2], 1.f, isclose);

    EXPECT_NEAR(z3[0], 1.f, isclose);
    EXPECT_NEAR(z3[1], 0.f, isclose);
    EXPECT_NEAR(z3[2], 0.f, isclose);
}

GTEST_TEST(detray_utils, euler_rotation2) {

    euler_rotation<algebra_t> euler_rot;
    euler_rot.alpha = constant<scalar_t>::pi / 6.f;
    euler_rot.beta = constant<scalar_t>::pi_4;
    euler_rot.gamma = constant<scalar_t>::pi_2;

    const test::vector3 v1{5.f, 7.f, 8.f};
    const test::vector3 v2 = euler_rot(v1);

    matrix_type<3u, 3u> R = matrix_operator().template zero<3u, 3u>();

    const scalar_t s1 = static_cast<scalar_t>(math::sin(euler_rot.alpha));
    const scalar_t c1 = static_cast<scalar_t>(math::cos(euler_rot.alpha));
    const scalar_t s2 = static_cast<scalar_t>(math::sin(euler_rot.beta));
    const scalar_t c2 = static_cast<scalar_t>(math::cos(euler_rot.beta));
    const scalar_t s3 = static_cast<scalar_t>(math::sin(euler_rot.gamma));
    const scalar_t c3 = static_cast<scalar_t>(math::cos(euler_rot.gamma));

    // From table of https://en.wikipedia.org/wiki/Euler_angles
    getter::element(R, 0u, 0u) = c1 * c3 - c2 * s1 * s3;
    getter::element(R, 0u, 1u) = -c1 * s3 - c2 * c3 * s1;
    getter::element(R, 0u, 2u) = s1 * s2;
    getter::element(R, 1u, 0u) = c3 * s1 + c1 * c2 * s3;
    getter::element(R, 1u, 1u) = c1 * c2 * c3 - s1 * s3;
    getter::element(R, 1u, 2u) = -c1 * s2;
    getter::element(R, 2u, 0u) = s2 * s3;
    getter::element(R, 2u, 1u) = c3 * s2;
    getter::element(R, 2u, 2u) = c2;

    const test::vector3 v3 = R * v1;

    EXPECT_NEAR(v2[0u], v3[0u], isclose);
    EXPECT_NEAR(v2[1u], v3[1u], isclose);
    EXPECT_NEAR(v2[2u], v3[2u], isclose);
}
