/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2020-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/geometry/shapes/single3D.hpp"

#include "detray/definitions/units.hpp"
#include "detray/geometry/mask.hpp"

// Detray test include(s)
#include "detray/test/utils/types.hpp"

// GTest include
#include <gtest/gtest.h>

using namespace detray;
using point3_t = test::point3;

constexpr scalar tol{1e-7f};

/// This tests the basic functionality of a single value mask (index 0)
GTEST_TEST(detray_masks, single3_0) {
    using point_t = point3_t;

    point_t p3_in = {0.5f, -9.f, 0.f};
    point_t p3_edge = {1.f, 9.3f, 2.f};
    point_t p3_out = {1.5f, -9.8f, 8.f};

    constexpr scalar h0{1.f * unit<scalar>::mm};
    mask<single3D<>> m1_0{0u, -h0, h0};

    ASSERT_NEAR(m1_0[single3D<>::e_lower], -h0, tol);
    ASSERT_NEAR(m1_0[single3D<>::e_upper], h0, tol);

    ASSERT_TRUE(m1_0.is_inside(p3_in));
    ASSERT_TRUE(m1_0.is_inside(p3_edge));
    ASSERT_FALSE(m1_0.is_inside(p3_out));
    // Move outside point inside using a tolerance - take t0 not t1
    ASSERT_TRUE(m1_0.is_inside(p3_out, 0.6f));

    // Check the measure
    EXPECT_NEAR(m1_0.measure(), 2.f * unit<scalar>::mm2, tol);

    // Check bounding box
    constexpr scalar envelope{0.01f};
    const auto loc_bounds = m1_0.local_min_bounds(envelope);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_min_x], -(h0 + envelope), tol);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_min_y], -envelope, tol);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_min_z], -envelope, tol);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_max_x], (h0 + envelope), tol);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_max_y], envelope, tol);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_max_z], envelope, tol);

    const auto centroid = m1_0.centroid();
    ASSERT_NEAR(centroid[0], 0.f, tol);
    ASSERT_NEAR(centroid[1], 0.f, tol);
    ASSERT_NEAR(centroid[2], 0.f, tol);
}

/// This tests the basic functionality of a single value mask (index 1)
GTEST_TEST(detray_masks, single3_1) {
    using point_t = point3_t;

    point_t p3_in = {0.5f, -9.f, 0.f};
    point_t p3_edge = {1.f, 9.3f, 2.f};
    point_t p3_out = {1.5f, -9.8f, 8.f};

    constexpr scalar h1{9.3f * unit<scalar>::mm};
    mask<single3D<1>> m1_1{0u, -h1, h1};

    ASSERT_NEAR(m1_1[single3D<>::e_lower], -h1, tol);
    ASSERT_NEAR(m1_1[single3D<>::e_upper], h1, tol);

    ASSERT_TRUE(m1_1.is_inside(p3_in));
    ASSERT_TRUE(m1_1.is_inside(p3_edge));
    ASSERT_FALSE(m1_1.is_inside(p3_out));
    // Move outside point inside using a tolerance - take t1 not t1
    ASSERT_TRUE(m1_1.is_inside(p3_out, 0.6f));

    // Check the measure
    EXPECT_NEAR(m1_1.measure(), 18.6f * unit<scalar>::mm2, tol);

    // Check bounding box
    constexpr scalar envelope{0.01f};
    const auto loc_bounds = m1_1.local_min_bounds(envelope);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_min_x], -envelope, tol);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_min_y], -(h1 + envelope), tol);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_min_z], -envelope, tol);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_max_x], envelope, tol);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_max_y], (h1 + envelope), tol);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_max_z], envelope, tol);

    const auto centroid = m1_1.centroid();
    ASSERT_NEAR(centroid[0], 0.f, tol);
    ASSERT_NEAR(centroid[1], 0.f, tol);
    ASSERT_NEAR(centroid[2], 0.f, tol);
}

/// This tests the basic functionality of a single value mask (index 2)
GTEST_TEST(detray_masks, single3_2) {
    using point_t = point3_t;

    point_t p3_in = {0.5f, -9.f, 0.f};
    point_t p3_edge = {1.f, 9.3f, 2.f};
    point_t p3_out = {1.5f, -9.8f, 8.f};

    constexpr scalar h2{2.f * unit<scalar>::mm};
    mask<single3D<2>> m1_2{0u, -h2, h2};

    ASSERT_NEAR(m1_2[single3D<>::e_lower], -h2, tol);
    ASSERT_NEAR(m1_2[single3D<>::e_upper], h2, tol);

    ASSERT_TRUE(m1_2.is_inside(p3_in));
    ASSERT_TRUE(m1_2.is_inside(p3_edge));
    ASSERT_FALSE(m1_2.is_inside(p3_out));
    // Move outside point inside using a tolerance - take t1 not t1
    ASSERT_TRUE(m1_2.is_inside(p3_out, 6.1f));

    // Check the measure
    EXPECT_NEAR(m1_2.measure(), 4.f * unit<scalar>::mm2, tol);

    // Check bounding box
    constexpr scalar envelope{0.01f};
    const auto loc_bounds = m1_2.local_min_bounds(envelope);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_min_x], -envelope, tol);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_min_y], -envelope, tol);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_min_z], -(h2 + envelope), tol);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_max_x], envelope, tol);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_max_y], envelope, tol);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_max_z], (h2 + envelope), tol);

    const auto centroid = m1_2.centroid();
    ASSERT_NEAR(centroid[0], 0.f, tol);
    ASSERT_NEAR(centroid[1], 0.f, tol);
    ASSERT_NEAR(centroid[2], 0.f, tol);
}
