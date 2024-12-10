/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "detray/utils/matrix_helper.hpp"

// Detray test include(s)
#include "detray/test/utils/types.hpp"

// GTest include(s)
#include <gtest/gtest.h>

using namespace detray;
using transform3 = test::transform3;
using vector3 = typename transform3::vector3;
using matrix_operator = test::matrix_operator;
template <std::size_t ROWS, std::size_t COLS>
using matrix_type = test::matrix<ROWS, COLS>;

constexpr scalar tolerance = 1e-6f;

GTEST_TEST(detray_utils, column_wise_cross) {
    auto P = matrix_operator().template zero<3, 3>();

    getter::element(P, 0u, 0u) = 0.f;
    getter::element(P, 0u, 1u) = 1.f;
    getter::element(P, 0u, 2u) = 2.f;
    getter::element(P, 1u, 0u) = 3.f;
    getter::element(P, 1u, 1u) = 4.f;
    getter::element(P, 1u, 2u) = 5.f;
    getter::element(P, 2u, 0u) = 6.f;
    getter::element(P, 2u, 1u) = 7.f;
    getter::element(P, 2u, 2u) = 8.f;

    const vector3 u{1.f, 2.f, 3.f};

    const auto Q = matrix_helper<matrix_operator>().column_wise_cross(P, u);

    EXPECT_NEAR(getter::element(Q, 0u, 0u), -3.f, tolerance);
    EXPECT_NEAR(getter::element(Q, 1u, 0u), 6.f, tolerance);
    EXPECT_NEAR(getter::element(Q, 2u, 0u), -3.f, tolerance);
    EXPECT_NEAR(getter::element(Q, 0u, 1u), -2.f, tolerance);
    EXPECT_NEAR(getter::element(Q, 1u, 1u), 4.f, tolerance);
    EXPECT_NEAR(getter::element(Q, 2u, 1u), -2.f, tolerance);
    EXPECT_NEAR(getter::element(Q, 0u, 2u), -1.f, tolerance);
    EXPECT_NEAR(getter::element(Q, 1u, 2u), 2.f, tolerance);
    EXPECT_NEAR(getter::element(Q, 2u, 2u), -1.f, tolerance);
}

GTEST_TEST(detray_utils, column_wise_multiply) {

    auto P = matrix_operator().template zero<3, 3>();

    getter::element(P, 0u, 0u) = 0.f;
    getter::element(P, 0u, 1u) = 1.f;
    getter::element(P, 0u, 2u) = 2.f;
    getter::element(P, 1u, 0u) = 3.f;
    getter::element(P, 1u, 1u) = 4.f;
    getter::element(P, 1u, 2u) = 5.f;
    getter::element(P, 2u, 0u) = 6.f;
    getter::element(P, 2u, 1u) = 7.f;
    getter::element(P, 2u, 2u) = 8.f;

    const vector3 u{1.f, 2.f, 3.f};

    const auto Q = matrix_helper<matrix_operator>().column_wise_multiply(P, u);

    EXPECT_NEAR(getter::element(Q, 0u, 0u), 0.f, tolerance);
    EXPECT_NEAR(getter::element(Q, 0u, 1u), 1.f, tolerance);
    EXPECT_NEAR(getter::element(Q, 0u, 2u), 2.f, tolerance);
    EXPECT_NEAR(getter::element(Q, 1u, 0u), 6.f, tolerance);
    EXPECT_NEAR(getter::element(Q, 1u, 1u), 8.f, tolerance);
    EXPECT_NEAR(getter::element(Q, 1u, 2u), 10.f, tolerance);
    EXPECT_NEAR(getter::element(Q, 2u, 0u), 18.f, tolerance);
    EXPECT_NEAR(getter::element(Q, 2u, 1u), 21.f, tolerance);
    EXPECT_NEAR(getter::element(Q, 2u, 2u), 24.f, tolerance);
}

GTEST_TEST(detray_utils, cross_matrix) {

    const vector3 u{1.f, 2.f, 3.f};
    const vector3 v{3.f, 4.f, 5.f};

    const auto u_cross = matrix_helper<matrix_operator>().cross_matrix(u);
    const auto v_cross = matrix_helper<matrix_operator>().cross_matrix(v);

    EXPECT_NEAR(getter::element(u_cross, 0u, 0u), 0.f, tolerance);
    EXPECT_NEAR(getter::element(u_cross, 0u, 1u), -3.f, tolerance);
    EXPECT_NEAR(getter::element(u_cross, 0u, 2u), 2.f, tolerance);
    EXPECT_NEAR(getter::element(u_cross, 1u, 0u), 3.f, tolerance);
    EXPECT_NEAR(getter::element(u_cross, 1u, 1u), 0.f, tolerance);
    EXPECT_NEAR(getter::element(u_cross, 1u, 2u), -1.f, tolerance);
    EXPECT_NEAR(getter::element(u_cross, 2u, 0u), -2.f, tolerance);
    EXPECT_NEAR(getter::element(u_cross, 2u, 1u), 1.f, tolerance);
    EXPECT_NEAR(getter::element(u_cross, 2u, 2u), 0.f, tolerance);

    // [u]_cross * v = [v]_cross^T * u
    const vector3 u_cross_v = u_cross * v;
    const vector3 v_cross_u = matrix_operator().transpose(v_cross) * u;

    EXPECT_NEAR(u_cross_v[0], -2.f, tolerance);
    EXPECT_NEAR(u_cross_v[1], 4.f, tolerance);
    EXPECT_NEAR(u_cross_v[2], -2.f, tolerance);
    EXPECT_NEAR(u_cross_v[0], v_cross_u[0], tolerance);
    EXPECT_NEAR(u_cross_v[1], v_cross_u[1], tolerance);
    EXPECT_NEAR(u_cross_v[2], v_cross_u[2], tolerance);
}

GTEST_TEST(detray_utils, outer_product) {

    const vector3 u{1.f, 2.f, 3.f};
    const vector3 v{3.f, 4.f, 5.f};

    const auto m33 = matrix_helper<matrix_operator>().outer_product(u, v);

    EXPECT_NEAR(getter::element(m33, 0u, 0u), 3.f, tolerance);
    EXPECT_NEAR(getter::element(m33, 0u, 1u), 4.f, tolerance);
    EXPECT_NEAR(getter::element(m33, 0u, 2u), 5.f, tolerance);
    EXPECT_NEAR(getter::element(m33, 1u, 0u), 6.f, tolerance);
    EXPECT_NEAR(getter::element(m33, 1u, 1u), 8.f, tolerance);
    EXPECT_NEAR(getter::element(m33, 1u, 2u), 10.f, tolerance);
    EXPECT_NEAR(getter::element(m33, 2u, 0u), 9.f, tolerance);
    EXPECT_NEAR(getter::element(m33, 2u, 1u), 12.f, tolerance);
    EXPECT_NEAR(getter::element(m33, 2u, 2u), 15.f, tolerance);
}

GTEST_TEST(detray_utils, cholesky_decomposition) {

    // Define A
    matrix_type<3, 3> A = matrix_operator().template zero<3, 3>();
    getter::element(A, 0u, 0u) = 4.f;
    getter::element(A, 0u, 1u) = 12.f;
    getter::element(A, 0u, 2u) = -16.f;
    getter::element(A, 1u, 0u) = 12.f;
    getter::element(A, 1u, 1u) = 37.f;
    getter::element(A, 1u, 2u) = -43.f;
    getter::element(A, 2u, 0u) = -16.f;
    getter::element(A, 2u, 1u) = -43.f;
    getter::element(A, 2u, 2u) = 98.f;

    // Get L that satisfies A = L * L^T and check if it is the expected value
    const matrix_type<3, 3> L =
        matrix_helper<matrix_operator>().cholesky_decompose(A);

    EXPECT_FLOAT_EQ(static_cast<float>(getter::element(L, 0u, 0u)), 2.f);
    EXPECT_FLOAT_EQ(static_cast<float>(getter::element(L, 0u, 1u)), 0.f);
    EXPECT_FLOAT_EQ(static_cast<float>(getter::element(L, 0u, 2u)), 0.f);
    EXPECT_FLOAT_EQ(static_cast<float>(getter::element(L, 1u, 0u)), 6.f);
    EXPECT_FLOAT_EQ(static_cast<float>(getter::element(L, 1u, 1u)), 1.f);
    EXPECT_FLOAT_EQ(static_cast<float>(getter::element(L, 1u, 2u)), 0.f);
    EXPECT_FLOAT_EQ(static_cast<float>(getter::element(L, 2u, 0u)), -8.f);
    EXPECT_FLOAT_EQ(static_cast<float>(getter::element(L, 2u, 1u)), 5.f);
    EXPECT_FLOAT_EQ(static_cast<float>(getter::element(L, 2u, 2u)), 3.f);

    // Compare A and L * L^T
    const matrix_type<3, 3> B = L * matrix_operator().transpose(L);

    for (unsigned int i = 0u; i < 3u; i++) {
        for (unsigned int j = 0u; j < 3u; j++) {
            EXPECT_FLOAT_EQ(static_cast<float>(getter::element(A, i, j)),
                            static_cast<float>(getter::element(B, i, j)));
        }
    }
}
