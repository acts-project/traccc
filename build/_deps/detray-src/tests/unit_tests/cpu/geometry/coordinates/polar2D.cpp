/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2020-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "detray/geometry/coordinates/polar2D.hpp"

#include "detray/definitions/units.hpp"

// Detray test include(s)
#include "detray/test/utils/types.hpp"

// GTest include(s).
#include <gtest/gtest.h>

using namespace detray;

using point3 = test::point3;
using vector3 = test::vector3;
using transform3 = test::transform3;

const scalar isclose{1e-5f};

// This test polar2D coordinate
GTEST_TEST(detray_coordinates, polar2D) {

    // Preparation work
    const vector3 z = {0.f, 0.f, 1.f};
    const vector3 x = {1.f, 0.f, 0.f};
    const point3 t = {2.f, 3.f, 4.f};
    const transform3 trf(t, z, x);
    const point3 global1 = {4.f, 7.f, 4.f};
    const vector3 mom = {1.f, 2.f, 3.f};
    const vector3 d = vector::normalize(mom);

    const polar2D<test::algebra> p2;

    // Global to local transformation
    const point3 local = p2.global_to_local_3D(trf, global1, d);

    // Check if the local position is correct
    ASSERT_NEAR(local[0], std::sqrt(20.f), isclose);
    ASSERT_NEAR(local[1], std::atan2(4.f, 2.f), isclose);

    // Local to global transformation
    const point3 global2 = p2.local_to_global(trf, local);

    // Check if the same global position is obtained
    ASSERT_NEAR(global1[0], global2[0], isclose);
    ASSERT_NEAR(global1[1], global2[1], isclose);
    ASSERT_NEAR(global1[2], global2[2], isclose);

    // Normal vector
    const vector3 n = p2.normal(trf);
    ASSERT_EQ(n, z);
}
