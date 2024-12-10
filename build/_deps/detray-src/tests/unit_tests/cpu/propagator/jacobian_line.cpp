/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2020-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "detray/propagator/detail/jacobian_line.hpp"

#include "detray/geometry/mask.hpp"
#include "detray/geometry/shapes/line.hpp"
#include "detray/propagator/detail/jacobian_engine.hpp"
#include "detray/tracks/tracks.hpp"

// Detray test include(s)
#include "detray/test/utils/types.hpp"

// GTest include(s).
#include <gtest/gtest.h>

using namespace detray;

using algebra_t = test::algebra;
using point2 = test::point2;
using point3 = test::point3;
using vector3 = test::vector3;
using transform3 = test::transform3;
using matrix_operator = test::matrix_operator;
template <std::size_t ROWS, std::size_t COLS>
using matrix_type = test::matrix<ROWS, COLS>;

constexpr scalar isclose{1e-5f};

const scalar r{2.f};
const scalar hz{50.f};
const mask<line<>> ln{0u, r, hz};

GTEST_TEST(detray_propagator, jacobian_line2D_case1) {

    using jac_engine = detail::jacobian_engine<line2D<algebra_t>>;

    // Preparation work
    vector3 z = {1.f, 1.f, 1.f};
    z = vector::normalize(z);
    vector3 x = {1.f, 0.f, -1.f};
    x = vector::normalize(x);
    const point3 t = {0.f, 0.f, 0.f};
    const transform3 trf(t, z, x);
    const point3 global1 = {1.f, 1.5f, 0.5f};
    const vector3 mom = {0.f, 1.f, 1.f};
    const scalar time{0.1f};
    const scalar charge{-1.f};

    // Free track parameter
    const free_track_parameters<algebra_t> free_params(global1, time, mom,
                                                       charge);

    const auto bound_vec =
        detail::free_to_bound_vector<line2D<algebra_t>>(trf, free_params);
    const auto free_params2 = detail::bound_to_free_vector(trf, ln, bound_vec);

    // Check if the bound vector is correct
    ASSERT_NEAR(bound_vec.bound_local()[e_bound_loc0],
                -constant<scalar>::inv_sqrt2, isclose);
    ASSERT_NEAR(bound_vec.bound_local()[e_bound_loc1], std::sqrt(3.f), isclose);
    ASSERT_NEAR(bound_vec.phi(), constant<scalar>::pi_2, isclose);  // atan(2)
    ASSERT_NEAR(bound_vec.theta(), constant<scalar>::pi_4,
                isclose);  // atan(sqrt(5)/3)
    ASSERT_NEAR(bound_vec.qop(), -constant<scalar>::inv_sqrt2, isclose);
    ASSERT_NEAR(bound_vec.time(), 0.1f, isclose);

    // Check if the same free vector is obtained
    for (unsigned int i = 0u; i < 8u; i++) {
        ASSERT_NEAR(free_params[i], free_params2[i], isclose);
    }

    // Test Jacobian transformation
    const bound_matrix<algebra_t> J =
        jac_engine::free_to_bound_jacobian(trf, free_params) *
        jac_engine::bound_to_free_jacobian(trf, ln, bound_vec);

    const matrix_operator m;

    for (unsigned int i = 0u; i < 6u; i++) {
        for (unsigned int j = 0u; j < 6u; j++) {
            if (i == j) {
                EXPECT_NEAR(m.element(J, i, j), 1.f, isclose);
            } else {
                EXPECT_NEAR(m.element(J, i, j), 0.f, isclose);
            }
        }
    }
}

GTEST_TEST(detray_coordinates, jacobian_line2D_case2) {

    using jac_engine = detail::jacobian_engine<line2D<algebra_t>>;

    // Preparation work
    vector3 z = {1.f, 2.f, 3.f};
    z = vector::normalize(z);
    vector3 x = {2.f, -4.f, 2.f};
    x = vector::normalize(x);
    const point3 t = {0.f, 0.f, 0.f};
    const transform3 trf(t, z, x);
    const vector3 mom = {1.f, 6.f, -2.f};
    const vector3 d = vector::normalize(mom);
    const scalar time{0.1f};
    const scalar charge{-1.f};

    const point2 bound1 = {1.f, 2.f};

    const point3 global =
        line2D<algebra_t>::local_to_global(trf, ln, bound1, d);

    // Free track parameter
    const free_track_parameters<algebra_t> free_params(global, time, mom,
                                                       charge);
    const auto bound_vec =
        detail::free_to_bound_vector<line2D<algebra_t>>(trf, free_params);

    // Test Jacobian transformation
    const bound_matrix<algebra_t> J =
        jac_engine::free_to_bound_jacobian(trf, free_params) *
        jac_engine::bound_to_free_jacobian(trf, ln, bound_vec);

    const matrix_operator m;

    for (unsigned int i = 0u; i < 6u; i++) {
        for (unsigned int j = 0u; j < 6u; j++) {
            if (i == j) {
                EXPECT_NEAR(m.element(J, i, j), 1.f, isclose);
            } else {
                EXPECT_NEAR(m.element(J, i, j), 0.f, isclose);
            }
        }
    }
}
