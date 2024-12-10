/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/geometry/detail/surface_descriptor.hpp"
#include "detray/geometry/mask.hpp"
#include "detray/geometry/shapes/line.hpp"
#include "detray/navigation/detail/ray.hpp"
#include "detray/navigation/intersection/intersection.hpp"
#include "detray/navigation/intersection/ray_intersector.hpp"
#include "detray/tracks/tracks.hpp"

// Detray test include(s)
#include "detray/test/utils/types.hpp"

// GTest include(s)
#include <gtest/gtest.h>

// System include(s)
#include <cmath>
#include <limits>

using namespace detray;

// Three-dimensional definitions
using algebra_t = test::algebra;
using transform3 = test::transform3;
using cartesian = cartesian2D<transform3>;
using vector3 = test::vector3;
using point3 = test::point3;
using point2 = test::point2;
using intersection_t = intersection2D<surface_descriptor<>, algebra_t, true>;
using line_intersector_type = ray_intersector<line_circular, algebra_t, true>;

constexpr scalar tol{1e-5f};

// Test simplest case
GTEST_TEST(detray_intersection, line_intersector_case1) {
    // tf3 with Identity rotation and no translation
    const transform3 tf{};

    // Create a track
    std::vector<free_track_parameters<algebra_t>> trks;
    trks.emplace_back(point3{1.f, -1.f, 0.f}, 0.f, vector3{0.f, 1.f, 0.f},
                      -1.f);
    trks.emplace_back(point3{-1.f, -1.f, 0.f}, 0.f, vector3{0.f, 1.f, 0.f},
                      -1.f);
    trks.emplace_back(point3{1.f, -1.f, 2.f}, 0.f, vector3{0.f, 1.f, -1.f},
                      -1.f);

    // Infinite wire with 10 mm radial cell size
    const mask<line_circular> ln{0u, 10.f,
                                 std::numeric_limits<scalar>::infinity()};

    // Test intersect
    std::vector<intersection_t> is(3u);
    is[0] = line_intersector_type()(detail::ray(trks[0]),
                                    surface_descriptor<>{}, ln, tf, tol);
    is[1] = line_intersector_type()(detail::ray(trks[1]),
                                    surface_descriptor<>{}, ln, tf, tol);
    is[2] = line_intersector_type()(detail::ray(trks[2]),
                                    surface_descriptor<>{}, ln, tf, tol);

    EXPECT_TRUE(is[0].status);
    EXPECT_EQ(is[0].path, 1.f);

    const auto global0 = ln.to_global_frame(tf, is[0].local);
    EXPECT_EQ(global0, point3({1.f, 0.f, 0.f}));
    EXPECT_EQ(is[0].local[0], -1.f);  // right
    EXPECT_EQ(is[0].local[1], 0.f);

    EXPECT_TRUE(is[1].status);
    EXPECT_EQ(is[1].path, 1.f);
    const auto global1 = ln.to_global_frame(tf, is[1].local);
    EXPECT_NEAR(global1[0], -1.f, tol);
    EXPECT_NEAR(global1[1], 0.f, tol);
    EXPECT_NEAR(global1[2], 0.f, tol);
    EXPECT_EQ(is[1].local[0], 1.f);  // left
    EXPECT_EQ(is[1].local[1], 0.f);

    EXPECT_TRUE(is[2].status);
    EXPECT_NEAR(is[2].path, constant<scalar>::sqrt2, tol);
    const auto global2 = ln.to_global_frame(tf, is[2].local);
    EXPECT_NEAR(global2[0], 1.f, tol);
    EXPECT_NEAR(global2[1], 0.f, tol);
    EXPECT_NEAR(global2[2], 1.f, tol);
    EXPECT_NEAR(is[2].local[0], -1.f, tol);  // right
    EXPECT_NEAR(is[2].local[1], 1.f, tol);
}

// Test inclined wire
GTEST_TEST(detray_intersection, line_intersector_case2) {
    // tf3 with skewed axis
    const vector3 x{1.f, 0.f, -1.f};
    const vector3 z{1.f, 0.f, 1.f};
    const vector3 t{1.f, 1.f, 1.f};
    const transform3 tf{t, vector::normalize(z), vector::normalize(x)};

    // Create a track
    const point3 pos{1.f, -1.f, 0.f};
    const vector3 dir{0.f, 1.f, 0.f};
    const free_track_parameters<algebra_t> trk(pos, 0.f, dir, -1.f);

    // Infinite wire with 10 mm
    // radial cell size
    const mask<line_circular> ln{0u, 10.f,
                                 std::numeric_limits<scalar>::infinity()};

    // Test intersect
    const intersection_t is = line_intersector_type()(
        detail::ray<algebra_t>(trk), surface_descriptor<>{}, ln, tf, tol);

    EXPECT_TRUE(is.status);
    EXPECT_NEAR(is.path, 2.f, tol);
    const auto global = ln.to_global_frame(tf, is.local);
    EXPECT_NEAR(global[0], 1.f, tol);
    EXPECT_NEAR(global[1], 1.f, tol);
    EXPECT_NEAR(global[2], 0.f, tol);
    EXPECT_NEAR(is.local[0], -constant<scalar>::inv_sqrt2, tol);  // right
    EXPECT_NEAR(is.local[1], -constant<scalar>::inv_sqrt2, tol);
}

GTEST_TEST(detray_intersection, line_intersector_square_scope) {

    // tf3 with Identity rotation and no translation
    const transform3 tf{};

    /// Create a track
    std::vector<free_track_parameters<algebra_t>> trks;
    trks.emplace_back(point3{2.f, 0.f, 0.f}, 0.f, vector3{-1.f, 1.f, 0.f},
                      -1.f);
    trks.emplace_back(point3{1.9f, 0.f, 0.f}, 0.f, vector3{-1.f, 1.f, 0.f},
                      -1.f);
    trks.emplace_back(point3{2.1f, 0.f, 0.f}, 0.f, vector3{-1.f, 1.f, 0.f},
                      -1.f);

    trks.emplace_back(point3{-2.f, 0.f, 0.f}, 0.f, vector3{1.f, 1.f, 0.f},
                      -1.f);
    trks.emplace_back(point3{-1.9f, 0.f, 0.f}, 0.f, vector3{1.f, 1.f, 0.f},
                      -1.f);
    trks.emplace_back(point3{-2.1f, 0.f, 0.f}, 0.f, vector3{1.f, 1.f, 0.f},
                      -1.f);

    trks.emplace_back(point3{0.f, -2.f, 0.f}, 0.f, vector3{-1.f, 1.f, 0.f},
                      -1.f);
    trks.emplace_back(point3{0.f, -1.9f, 0.f}, 0.f, vector3{-1.f, 1.f, 0.f},
                      -1.f);
    trks.emplace_back(point3{0.f, -2.1f, 0.f}, 0.f, vector3{-1.f, 1.f, 0.f},
                      -1.f);

    trks.emplace_back(point3{0.f, -2.f, 0.f}, 0.f, vector3{1.f, 1.f, 0.f},
                      -1.f);
    trks.emplace_back(point3{0.f, -1.9f, 0.f}, 0.f, vector3{1.f, 1.f, 0.f},
                      -1.f);
    trks.emplace_back(point3{0.f, -2.1f, 0.f}, 0.f, vector3{1.f, 1.f, 0.f},
                      -1.f);

    // Infinite wire with 1 mm square cell size
    mask<line_square, std::uint_least16_t, algebra_t> ln{
        0u, 1.f, std::numeric_limits<scalar>::infinity()};

    // Test intersect
    std::vector<intersection_t> is;
    for (const auto& trk : trks) {
        is.push_back(line_intersector_type()(
            detail::ray<algebra_t>(trk), surface_descriptor<>{}, ln, tf, tol));
    }

    EXPECT_TRUE(is[0].status);
    EXPECT_NEAR(is[0].path, constant<scalar>::sqrt2, tol);
    const auto local0 = ln.to_local_frame(
        tf,
        detail::ray(trks[0]).pos() + is[0].path * detail::ray(trks[0]).dir(),
        detail::ray(trks[0]).dir());
    const auto global0 = ln.to_global_frame(tf, local0);
    EXPECT_NEAR(global0[0], 1.f, tol);
    EXPECT_NEAR(global0[1], 1.f, tol);
    EXPECT_NEAR(global0[2], 0.f, tol);
    EXPECT_NEAR(is[0].local[0], -constant<scalar>::sqrt2, tol);
    EXPECT_NEAR(is[0].local[1], 0.f, tol);

    EXPECT_TRUE(is[1].status);
    EXPECT_TRUE(std::signbit(is[1].local[0]));
    EXPECT_FALSE(is[2].status);
    EXPECT_TRUE(std::signbit(is[2].local[0]));

    EXPECT_TRUE(is[3].status);
    EXPECT_FALSE(std::signbit(is[3].local[0]));
    EXPECT_TRUE(is[4].status);
    EXPECT_FALSE(std::signbit(is[4].local[0]));
    EXPECT_FALSE(is[5].status);
    EXPECT_FALSE(std::signbit(is[5].local[0]));

    EXPECT_TRUE(is[6].status);
    EXPECT_FALSE(std::signbit(is[6].local[0]));
    EXPECT_TRUE(is[7].status);
    EXPECT_FALSE(std::signbit(is[7].local[0]));
    EXPECT_FALSE(is[8].status);
    EXPECT_FALSE(std::signbit(is[8].local[0]));

    EXPECT_TRUE(is[9].status);
    EXPECT_TRUE(std::signbit(is[9].local[0]));
    EXPECT_TRUE(is[10].status);
    EXPECT_TRUE(std::signbit(is[10].local[0]));
    EXPECT_FALSE(is[11].status);
    EXPECT_TRUE(std::signbit(is[11].local[0]));
}
