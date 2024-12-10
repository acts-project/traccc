/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "detray/navigation/intersection/helix_intersector.hpp"

#include "detray/definitions/units.hpp"
#include "detray/geometry/detail/surface_descriptor.hpp"
#include "detray/geometry/mask.hpp"
#include "detray/geometry/shapes/concentric_cylinder2D.hpp"
#include "detray/geometry/shapes/line.hpp"
#include "detray/geometry/shapes/rectangle2D.hpp"
#include "detray/geometry/shapes/unmasked.hpp"
#include "detray/navigation/detail/helix.hpp"
#include "detray/tracks/tracks.hpp"

// Detray test include(s)
#include "detray/test/utils/types.hpp"

// Google Test include(s).
#include <gtest/gtest.h>

// System include(s)
#include <cmath>
#include <limits>

using namespace detray;

namespace {

// Three-dimensional definitions
using algebra_t = test::algebra;
using transform3_t = test::transform3;
using vector3 = test::vector3;
using point3 = test::point3;
using helix_t = detray::detail::helix<algebra_t>;
using intersection_t = intersection2D<surface_descriptor<>, algebra_t, true>;

constexpr auto not_defined{detail::invalid_value<scalar>()};
constexpr scalar tol{1e-4f};

// z axis
const vector3 z_axis{0.f, 0.f, 1.f};

// Track defined on origin point
const free_track_parameters<algebra_t> free_trk(
    {0.f, 0.f, 0.f}, 0.f, {0.1f * unit<scalar>::GeV, 0.f, 0.f}, -1.f);

// Magnetic field
const vector3 B{0.f, 0.f, 1.f * unit<scalar>::T};
const vector3 B_0{0.f * unit<scalar>::T, tol* unit<scalar>::T,
                  tol* unit<scalar>::T};

// Test helix
const helix_t hlx(free_trk, &B);

// Path along the helix
const scalar path = 10.f * unit<scalar>::cm;

// Transform translation vector
const vector3 trl = hlx(path);

// Surface normal vector
const vector3 w = hlx.dir(path);

}  // anonymous namespace

/// This defines the local frame test suite
GTEST_TEST(detray_intersection, helix_plane_intersector_no_bfield) {
    // Create a shifted plane
    const transform3_t shifted(vector3{3.f, 2.f, 10.f});

    // Test helix
    const point3 pos{2.f, 1.f, 0.f};
    const vector3 mom{0.f, 0.f, 1.f};
    const detail::helix<algebra_t> h({pos, 0.f, mom, -1.f}, &B_0);

    // The same test but bound to local frame
    helix_intersector<unmasked<2>, algebra_t> pi;
    mask<unmasked<2>> unmasked_bound{};
    const auto hit_bound =
        pi(h, surface_descriptor<>{}, unmasked_bound, shifted, tol);

    ASSERT_TRUE(hit_bound.status);
    // Global intersection information - unchanged
    const auto global0 =
        unmasked_bound.to_global_frame(shifted, hit_bound.local);
    ASSERT_NEAR(global0[0], 2.f, tol);
    ASSERT_NEAR(global0[1], 1.f, tol);
    ASSERT_NEAR(global0[2], 10.f, tol);
    // Local intersection information
    ASSERT_NEAR(hit_bound.local[0], -1.f, tol);
    ASSERT_NEAR(hit_bound.local[1], -1.f, tol);

    // The same test but bound to local frame & masked - inside
    mask<rectangle2D> rect_for_inside{0u, 3.f, 3.f};
    const auto hit_bound_inside =
        pi(h, surface_descriptor<>{}, rect_for_inside, shifted, tol);
    ASSERT_TRUE(hit_bound_inside.status);
    // Global intersection information - unchanged
    const auto global1 =
        rect_for_inside.to_global_frame(shifted, hit_bound_inside.local);
    ASSERT_NEAR(global1[0], 2.f, tol);
    ASSERT_NEAR(global1[1], 1.f, tol);
    ASSERT_NEAR(global1[2], 10.f, tol);
    // Local intersection infoimation - unchanged
    ASSERT_NEAR(hit_bound_inside.local[0], -1.f, tol);
    ASSERT_NEAR(hit_bound_inside.local[1], -1.f, tol);

    // The same test but bound to local frame & masked - outside
    mask<rectangle2D> rect_for_outside{0u, 0.5f, 3.5f};
    const auto hit_bound_outside =
        pi(h, surface_descriptor<>{}, rect_for_outside, shifted, tol);
    ASSERT_FALSE(hit_bound_outside.status);
    const auto global2 =
        rect_for_outside.to_global_frame(shifted, hit_bound_outside.local);
    // Global intersection information - unchanged
    ASSERT_NEAR(global2[0], 2.f, tol);
    ASSERT_NEAR(global2[1], 1.f, tol);
    ASSERT_NEAR(global2[2], 10.f, tol);
    // Local intersection infoimation - unchanged
    ASSERT_NEAR(hit_bound_outside.local[0], -1.f, tol);
    ASSERT_NEAR(hit_bound_outside.local[1], -1.f, tol);
}

/// Test the intersection between a helical trajectory and a plane
GTEST_TEST(detray_intersection, helix_plane_intersector) {

    // Vector on the surface
    const vector3 v = vector::cross(z_axis, w);

    // Transform matrix
    const transform3_t trf(trl, w, v);

    // Rectangle surface
    const mask<rectangle2D> rectangle{0u, 10.f * unit<scalar>::cm,
                                      10.f * unit<scalar>::cm};

    const helix_intersector<rectangle2D, algebra_t> hpi;

    // Get the intersection on the next surface
    const auto is = hpi(hlx, surface_descriptor<>{}, rectangle, trf, tol);

    // Check the values
    EXPECT_TRUE(is.status);
    EXPECT_NEAR(is.path, path, tol);
    EXPECT_NEAR(is.local[0], 0.f, tol);
    EXPECT_NEAR(is.local[1], 0.f, tol);

    const auto global = rectangle.to_global_frame(trf, is.local);
    EXPECT_NEAR(global[0], trl[0], tol);
    EXPECT_NEAR(global[1], trl[1], tol);
    EXPECT_NEAR(global[2], trl[2], tol);
}

/// This checks the closest solution of a helix-cylinder intersection
GTEST_TEST(detray_intersection, helix_cylinder_intersector_no_bfield) {

    const scalar r{4.f * unit<scalar>::mm};
    const scalar hz{10.f * unit<scalar>::mm};

    // Create a translated cylinder and test untersection
    const transform3_t shifted(vector3{3.f, 2.f, 10.f});
    helix_intersector<cylinder2D, algebra_t> hi;

    // Test helix
    const point3 pos{3.f, 2.f, 5.f};
    const vector3 mom{1.f, 0.f, 0.f};
    const helix_t h({pos, 0.f * unit<scalar>::s, mom, -1 * unit<scalar>::e},
                    &B_0);

    // Intersect
    mask<cylinder2D, std::uint_least16_t, algebra_t> cylinder{0u, r, -hz, hz};
    const auto hits_bound =
        hi(h, surface_descriptor<>{}, cylinder, shifted, tol);

    // No magnetic field, so the solutions must be the same as for a ray

    // second intersection lies in front of the track
    EXPECT_TRUE(hits_bound[0].status);
    EXPECT_FALSE(hits_bound[0].direction);

    const auto global0 = cylinder.to_global_frame(shifted, hits_bound[0].local);

    EXPECT_NEAR(global0[0], -1.f, tol);
    EXPECT_NEAR(global0[1], 2.f, tol);
    EXPECT_NEAR(global0[2], 5.f, tol);
    ASSERT_TRUE(hits_bound[0].local[0] != not_defined &&
                hits_bound[0].local[1] != not_defined);
    // p2[0] = r * phi : 180deg in the opposite direction with r = 4
    EXPECT_NEAR(hits_bound[0].local[0], 4.f * M_PI, tol);
    EXPECT_NEAR(hits_bound[0].local[1], -5.f, tol);

    // first intersection lies behind the track
    const auto global1 = cylinder.to_global_frame(shifted, hits_bound[1].local);
    EXPECT_TRUE(hits_bound[1].status);
    EXPECT_TRUE(hits_bound[1].direction);
    EXPECT_NEAR(global1[0], 7.f, tol);
    EXPECT_NEAR(global1[1], 2.f, tol);
    EXPECT_NEAR(global1[2], 5.f, tol);
    ASSERT_TRUE(hits_bound[1].local[0] != not_defined &&
                hits_bound[1].local[1] != not_defined);
    EXPECT_NEAR(hits_bound[1].local[0], 0.f, tol);
    EXPECT_NEAR(hits_bound[1].local[1], -5., tol);
}

/// Test the intersection between a helical trajectory and a cylinder
GTEST_TEST(detray_intersection, helix_cylinder_intersector) {

    // Transform matrix
    const transform3_t trf(trl, z_axis, w);

    // Cylinder surface (5 cm radius)
    const scalar r{4.f * unit<scalar>::cm};
    const scalar hz{10.f * unit<scalar>::cm};
    const mask<cylinder2D> cylinder{0u, r, -hz, hz};

    const helix_intersector<cylinder2D, algebra_t> hci;

    // Get the intersection on the next surface
    const auto is = hci(hlx, surface_descriptor<>{}, cylinder, trf, tol);

    // First solution
    const vector3 pos_near = hlx.pos(is[0].path);
    const vector3 loc_near = pos_near - trl;
    const scalar phi_near = std::acos(
        vector::dot(w, loc_near) / (getter::norm(w) * getter::norm(loc_near)));

    EXPECT_TRUE(is[0].status);
    // Not precise due to helix curvature
    EXPECT_NEAR(is[0].path, path - r, 5000.f * tol);
    EXPECT_NEAR(is[0].local[0], r * phi_near, tol);
    EXPECT_NEAR(is[0].local[1], 0.f, tol);
    const auto global0 = cylinder.to_global_frame(trf, is[0].local);
    EXPECT_NEAR(global0[0], pos_near[0], tol);
    EXPECT_NEAR(global0[1], pos_near[1], tol);
    EXPECT_NEAR(global0[2], pos_near[2], tol);

    // Second solution
    const vector3 pos_far = hlx.pos(is[1].path);
    const vector3 loc_far = pos_far - trl;
    const scalar phi_far = std::acos(vector::dot(w, loc_far) /
                                     (getter::norm(w) * getter::norm(loc_far)));

    EXPECT_TRUE(is[1].status);
    // Not precise due to helix curvature
    EXPECT_NEAR(is[1].path, path + r, 5000.f * tol);
    EXPECT_NEAR(is[1].local[0], r * phi_far, tol);
    EXPECT_NEAR(is[1].local[1], 0.f, tol);
    const auto global1 = cylinder.to_global_frame(trf, is[1].local);
    EXPECT_NEAR(global1[0], pos_far[0], tol);
    EXPECT_NEAR(global1[1], pos_far[1], tol);
    EXPECT_NEAR(global1[2], pos_far[2], tol);
}

/// This checks the closest solution of a helix-concentric cylinder intersection
GTEST_TEST(detray_intersection,
           helix_concentric_cylinder_intersector_no_bfield) {

    const scalar r{4.f * unit<scalar>::mm};
    const scalar hz{10.f * unit<scalar>::mm};

    // Create a translated cylinder and test untersection
    const transform3_t identity{};
    helix_intersector<concentric_cylinder2D, algebra_t> c_hi;

    // Test helix
    const point3 pos{0.f, 0.f, -5.f};
    const vector3 mom{1.f, 0.f, 0.f};
    const helix_t h({pos, 0.f * unit<scalar>::s, mom, -1 * unit<scalar>::e},
                    &B_0);

    // Intersect
    mask<concentric_cylinder2D, std::uint_least16_t, algebra_t> cylinder{
        0u, r, -hz, hz};
    const auto hits_bound =
        c_hi(h, surface_descriptor<>{}, cylinder, identity, tol);

    // No magnetic field, so the solutions must be the same as for a ray

    // second intersection lies in front of the track
    EXPECT_TRUE(hits_bound[0].status);
    EXPECT_FALSE(hits_bound[0].direction);

    const auto global0 =
        cylinder.to_global_frame(identity, hits_bound[0].local);

    EXPECT_NEAR(global0[0], -4.f, tol);
    EXPECT_NEAR(global0[1], 0.f, tol);
    EXPECT_NEAR(global0[2], -5.f, tol);
    ASSERT_TRUE(hits_bound[0].local[0] != not_defined &&
                hits_bound[0].local[1] != not_defined);
    // p2[0] = phi : 180deg in the opposite direction
    EXPECT_NEAR(hits_bound[0].local[0], constant<scalar>::pi, tol);
    EXPECT_NEAR(hits_bound[0].local[1], -5.f, tol);

    // first intersection lies behind the track
    const auto global1 =
        cylinder.to_global_frame(identity, hits_bound[1].local);
    EXPECT_TRUE(hits_bound[1].status);
    EXPECT_TRUE(hits_bound[1].direction);
    EXPECT_NEAR(global1[0], 4.f, tol);
    EXPECT_NEAR(global1[1], 0.f, tol);
    EXPECT_NEAR(global1[2], -5.f, tol);
    ASSERT_TRUE(hits_bound[1].local[0] != not_defined &&
                hits_bound[1].local[1] != not_defined);
    EXPECT_NEAR(hits_bound[1].local[0], 0.f, tol);
    EXPECT_NEAR(hits_bound[1].local[1], -5.f, tol);
}

/// Test the intersection between a helical trajectory and a line
GTEST_TEST(detray_intersection, helix_line_intersector) {

    // Intersector object
    const helix_intersector<line_circular, algebra_t> hli;

    // Get radius of track
    const scalar R{hlx.radius()};

    // Path length for pi/4
    const scalar s0 = constant<scalar>::pi_2 * R;

    // Wire properties
    const scalar scope = 2.f * unit<scalar>::cm;
    const scalar half_z = std::numeric_limits<scalar>::max();

    // Straw wire
    const mask<line_circular> straw_tube{0u, scope, half_z};

    // Cell wire
    const mask<line_square> drift_cell{0u, scope, half_z};

    // Offset to shift the translation of transform matrix
    const scalar offset = 1.f * unit<scalar>::cm;

    //---------------------
    // Forward direction
    //---------------------

    // Reference point for transform matrix
    const point3 r0_fw = hlx.pos(s0);

    // Translation is shifted from reference point
    const point3 trl_fw = r0_fw + vector3{offset, 0.f, 0.f};

    // Transform matrix
    const transform3_t trf_fw(trl_fw, z_axis, hlx.dir(s0));

    // Get the intersection on the next surface
    auto is = hli(hlx, surface_descriptor<>{}, straw_tube, trf_fw, tol);

    EXPECT_NEAR(is.path, s0, tol);
    // track (helix) is at the left side w.r.t wire
    EXPECT_NEAR(is.local[0], offset, tol);
    EXPECT_NEAR(is.local[1], 0.f, tol);
    EXPECT_TRUE(is.status);
    EXPECT_TRUE(is.direction);

    // Get the intersection on the next surface
    is = hli(hlx, surface_descriptor<>{}, drift_cell, trf_fw, tol);

    EXPECT_NEAR(is.path, s0, tol);
    // track (helix) is at the left side w.r.t wire
    EXPECT_NEAR(is.local[0], offset, tol);
    EXPECT_NEAR(is.local[1], 0.f, tol);
    EXPECT_TRUE(is.status);
    EXPECT_TRUE(is.direction);

    //---------------------
    // Backward direction
    //---------------------

    // Reference point for transform matrix
    const point3 r0_bw = hlx.pos(-s0);

    // Translation is shifted from reference point
    const point3 trl_bw = r0_bw + vector3{offset, 0.f, 0.f};

    // Transform matrix
    const transform3_t trf_bw(trl_bw, z_axis, hlx.dir(-s0));

    // Get the intersection on the next surface
    is = hli(hlx, surface_descriptor<>{}, straw_tube, trf_bw, tol);

    EXPECT_NEAR(is.path, -s0, tol);
    // track (helix) is at the right side w.r.t wire
    EXPECT_NEAR(is.local[0], -offset, tol);
    EXPECT_NEAR(is.local[1], 0.f, tol);
    EXPECT_TRUE(is.status);
    EXPECT_FALSE(is.direction);

    // Get the intersection on the next surface
    is = hli(hlx, surface_descriptor<>{}, drift_cell, trf_bw, tol);

    EXPECT_NEAR(is.path, -s0, tol);
    // track (helix) is at the right side w.r.t wire
    EXPECT_NEAR(is.local[0], -offset, tol);
    EXPECT_NEAR(is.local[1], 0.f, tol);
    EXPECT_TRUE(is.status);
    EXPECT_FALSE(is.direction);
}
