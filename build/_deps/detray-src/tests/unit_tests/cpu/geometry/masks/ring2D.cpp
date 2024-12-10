/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2020-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/geometry/shapes/ring2D.hpp"

#include "detray/definitions/units.hpp"
#include "detray/geometry/mask.hpp"

// Detray test include(s)
#include "detray/test/utils/ratio_test.hpp"
#include "detray/test/utils/types.hpp"

// GTest include
#include <gtest/gtest.h>

using namespace detray;
using point3_t = test::point3;

constexpr scalar tol{1e-5f};

/// This tests the basic functionality of a ring
GTEST_TEST(detray_masks, ring2D) {
    using point_t = point3_t;

    point_t p2_pl_in = {0.5f, -2.f, 0.f};
    point_t p2_pl_edge = {0.f, 3.5f, 0.f};
    point_t p2_pl_out = {3.6f, 5.f, 0.f};

    constexpr scalar inner_r{0.f * unit<scalar>::mm};
    constexpr scalar outer_r{3.5f * unit<scalar>::mm};

    mask<ring2D> r2{0u, inner_r, outer_r};

    ASSERT_NEAR(r2[ring2D::e_inner_r], 0.f, tol);
    ASSERT_NEAR(r2[ring2D::e_outer_r], 3.5f, tol);

    ASSERT_TRUE(r2.is_inside(p2_pl_in));
    ASSERT_TRUE(r2.is_inside(p2_pl_edge));
    ASSERT_FALSE(r2.is_inside(p2_pl_out));
    // Move outside point inside using a tolerance
    ASSERT_TRUE(r2.is_inside(p2_pl_out, 1.2f));

    // Check area
    const scalar a{r2.area()};
    EXPECT_NEAR(a, 38.4845100065f * unit<scalar>::mm2, tol);
    ASSERT_EQ(a, r2.measure());

    // Check bounding box
    constexpr scalar envelope{0.01f};
    const auto loc_bounds = r2.local_min_bounds(envelope);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_min_x], -(outer_r + envelope), tol);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_min_y], -(outer_r + envelope), tol);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_min_z], -envelope, tol);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_max_x], (outer_r + envelope), tol);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_max_y], (outer_r + envelope), tol);
    ASSERT_NEAR(loc_bounds[cuboid3D::e_max_z], envelope, tol);

    const auto centroid = r2.centroid();
    ASSERT_NEAR(centroid[0], 0.f, tol);
    ASSERT_NEAR(centroid[1], 0.f, tol);
    ASSERT_NEAR(centroid[2], 0.f, tol);
}

/// This tests the inside/outside method of the mask
GTEST_TEST(detray_masks, ring2D_ratio_test) {

    struct mask_check {
        bool operator()(const test::point3 &p, const mask<ring2D> &r,
                        const test::transform3 &trf, const scalar t) {

            const test::point3 loc_p{r.to_local_frame(trf, p)};
            return r.is_inside(loc_p, t);
        }
    };

    constexpr mask<ring2D> r{0u, 2.f, 5.f};

    constexpr scalar t{0.f};
    const test::transform3 trf{};
    constexpr scalar size{10.f * unit<scalar>::mm};
    const auto n_points{static_cast<std::size_t>(std::pow(500, 3))};

    // x- and y-coordinates yield a valid local position on the underlying plane
    std::vector<test::point3> points =
        test::generate_regular_points<cuboid3D>(n_points, {size});

    scalar ratio = test::ratio_test<mask_check>(points, r, trf, t);

    const scalar area{r.measure()};
    const scalar world{size * size};

    ASSERT_NEAR(ratio, area / world, 0.001f);
}
