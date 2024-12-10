/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2020-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "detray/propagator/detail/jacobian_cartesian.hpp"

#include "detray/geometry/mask.hpp"
#include "detray/geometry/shapes/rectangle2D.hpp"
#include "detray/propagator/detail/jacobian_engine.hpp"
#include "detray/tracks/tracks.hpp"

// Detray test include(s)
#include "detray/test/utils/types.hpp"

// GTest include(s).
#include <gtest/gtest.h>

using namespace detray;

using algebra_t = test::algebra;
using point3 = test::point3;
using vector3 = test::vector3;
using transform3 = test::transform3;
using matrix_operator = test::matrix_operator;
template <std::size_t ROWS, std::size_t COLS>
using matrix_type = test::matrix<ROWS, COLS>;

const scalar isclose{1e-5f};

// This test cartesian2D coordinate
GTEST_TEST(detray_propagator, jacobian_cartesian2D) {

    using jac_engine = detail::jacobian_engine<cartesian2D<algebra_t>>;

    // Preparation work
    const vector3 z = {0.f, 0.f, 1.f};
    const vector3 x = {1.f, 0.f, 0.f};
    const point3 t = {2.f, 3.f, 4.f};
    const transform3 trf(t, z, x);
    const point3 global1 = {4.f, 7.f, 4.f};
    const vector3 mom = {1.f, 2.f, 3.f};
    const scalar time{0.1f};
    const scalar charge{-1.f};

    const scalar h{2.f};
    mask<rectangle2D> rect{0u, h, h};

    // Free track parameter
    const free_track_parameters<algebra_t> free_params(global1, time, mom,
                                                       charge);

    const auto bound_vec =
        detail::free_to_bound_vector<cartesian2D<algebra_t>>(trf, free_params);
    const auto free_params2 =
        detail::bound_to_free_vector(trf, rect, bound_vec);

    // Check if the bound vector is correct
    ASSERT_NEAR(bound_vec.bound_local()[e_bound_loc0], 2.f, isclose);
    ASSERT_NEAR(bound_vec.bound_local()[e_bound_loc1], 4.f, isclose);
    ASSERT_NEAR(bound_vec.phi(), 1.1071487f, isclose);     // atan(2)
    ASSERT_NEAR(bound_vec.theta(), 0.64052231f, isclose);  // atan(sqrt(5)/3)
    ASSERT_NEAR(bound_vec.qop(), -1.f / 3.7416574f, isclose);
    ASSERT_NEAR(bound_vec.time(), 0.1f, isclose);

    // Check if the same free vector is obtained
    for (unsigned int i = 0u; i < 8u; i++) {
        ASSERT_NEAR(free_params[i], free_params2[i], isclose);
    }

    // Test Jacobian transformation
    const bound_matrix<algebra_t> J =
        jac_engine::free_to_bound_jacobian(trf, free_params) *
        jac_engine::bound_to_free_jacobian(trf, rect, bound_vec);

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
