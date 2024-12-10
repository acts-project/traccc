/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/geometry/tracking_surface.hpp"

#include "detray/definitions/detail/indexing.hpp"

// Detray test include(s)
#include "detray/test/utils/detectors/build_toy_detector.hpp"
#include "detray/test/utils/detectors/build_wire_chamber.hpp"
#include "detray/test/utils/types.hpp"

// Vecmem include(s)
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s)
#include <gtest/gtest.h>

namespace {

/// Define mask types
enum class mask_ids : unsigned int {
    e_unmasked = 0u,
};

/// Define material types
enum class material_ids : unsigned int {
    e_slab = 0u,
};

constexpr detray::scalar tol{5e-5f};

}  // anonymous namespace

// This tests the construction of a surface descriptor object
GTEST_TEST(detray_geometry, surface_descriptor) {

    using namespace detray;

    using mask_link_t = dtyped_index<mask_ids, dindex>;
    using material_link_t = dtyped_index<material_ids, dindex>;

    mask_link_t mask_id{mask_ids::e_unmasked, 0u};
    material_link_t material_id{material_ids::e_slab, 0u};

    surface_descriptor<mask_link_t, material_link_t> desc(
        1u, mask_id, material_id, 2u, surface_id::e_sensitive);

    static_assert(sizeof(decltype(desc)) == 16);

    // Test access
    ASSERT_EQ(desc.transform(), 1u);
    ASSERT_EQ(desc.volume(), 2u);
    ASSERT_EQ(desc.id(), surface_id::e_sensitive);
    ASSERT_FALSE(desc.is_portal());
    ASSERT_FALSE(desc.is_passive());
    ASSERT_TRUE(desc.is_sensitive());
    ASSERT_EQ(desc.mask(), mask_id);
    ASSERT_EQ(desc.material(), material_id);

    // Test setters
    desc.set_volume(5u);
    desc.set_id(surface_id::e_portal);
    desc.set_index(6u);
    desc.update_transform(7u);
    desc.update_mask(7u);
    desc.update_material(7u);

    ASSERT_EQ(desc.transform(), 8u);
    ASSERT_EQ(desc.volume(), 5u);
    ASSERT_EQ(desc.id(), surface_id::e_portal);
    ASSERT_EQ(desc.index(), 6u);
    ASSERT_TRUE(desc.is_portal());
    ASSERT_FALSE(desc.is_passive());
    ASSERT_FALSE(desc.is_sensitive());
    ASSERT_EQ(desc.mask().id(), mask_ids::e_unmasked);
    ASSERT_EQ(desc.mask().index(), 7u);
    ASSERT_EQ(desc.material().id(), material_ids::e_slab);
    ASSERT_EQ(desc.material().index(), 7u);
}

/// This tests the functionality of a tracking surface interface for the toy
/// detector
GTEST_TEST(detray_geometry, surface_toy_detector) {

    using namespace detray;

    using detector_t = detector<toy_metadata>;

    using scalar_t = tracking_surface<detector_t>::scalar_type;
    using point2_t = tracking_surface<detector_t>::point2_type;
    using point3_t = tracking_surface<detector_t>::point3_type;
    using vector3_t = tracking_surface<detector_t>::vector3_type;

    vecmem::host_memory_resource host_mr;
    toy_det_config toy_cfg{};
    toy_cfg.use_material_maps(true).do_check(true);
    const auto [toy_det, names] = build_toy_detector(host_mr, toy_cfg);

    auto ctx = typename detector_t::geometry_context{};

    //
    // Disc
    //
    const auto disc_descr = toy_det.surfaces()[1u];
    const auto disc = tracking_surface{toy_det, disc_descr};

    // IDs
    ASSERT_EQ(disc.barcode(), disc_descr.barcode());
    ASSERT_EQ(disc.volume(), 0u);
    ASSERT_EQ(disc.index(), 1u);
    ASSERT_EQ(disc.id(), surface_id::e_portal);
    ASSERT_EQ(disc.shape_id(), detector_t::masks::id::e_portal_ring2);
    ASSERT_EQ(disc.shape_name(), "ring2D");
    ASSERT_FALSE(disc.is_sensitive());
    ASSERT_FALSE(disc.is_passive());
    ASSERT_TRUE(disc.is_portal());

    // Transformation matrix
    const auto disc_translation =
        disc.transform(ctx).translation();  // beampipe portal
    ASSERT_NEAR(disc_translation[0], 0.f, tol);
    ASSERT_NEAR(disc_translation[1], 0.f, tol);
    ASSERT_NEAR(disc_translation[2], -824.5f, tol);
    auto center = disc.center(ctx);
    ASSERT_NEAR(center[0], 0.f, tol);
    ASSERT_NEAR(center[1], 0.f, tol);
    ASSERT_NEAR(center[2], -824.5f, tol);

    // Surface normal
    const auto z_axis = vector3_t{0.f, 0.f, 1.f};
    // trigger all code paths
    ASSERT_EQ(disc.normal(ctx, point3_t{0.f, 0.f, 0.f}), z_axis);
    ASSERT_EQ(disc.normal(ctx, point2_t{0.f, 0.f}), z_axis);

    // Cos incidence angle
    auto dir = vector::normalize(vector3_t{1.f, 0.f, 1.f});
    ASSERT_NEAR(disc.cos_angle(ctx, dir, point3_t{0.f, 0.f, 0.f}),
                constant<scalar_t>::inv_sqrt2, tol);
    ASSERT_NEAR(disc.cos_angle(ctx, dir, point2_t{0.f, 0.f}),
                constant<scalar_t>::inv_sqrt2, tol);

    dir = vector::normalize(vector3_t{1.f, 1.f, 0.f});
    ASSERT_NEAR(disc.cos_angle(ctx, dir, point3_t{1.f, 0.5f, 0.f}), 0.f, tol);
    ASSERT_NEAR(disc.cos_angle(ctx, dir, point2_t{1.f, 0.5f}), 0.f, tol);

    dir = vector::normalize(vector3_t{1.f, 0.f, constant<scalar_t>::pi});
    scalar_t cos_inc_angle{
        constant<scalar_t>::pi /
        std::sqrt(1.f + std::pow(constant<scalar_t>::pi, 2.f))};
    ASSERT_NEAR(disc.cos_angle(ctx, dir, point3_t{2.f, 1.f, 0.f}),
                cos_inc_angle, tol);
    ASSERT_NEAR(disc.cos_angle(ctx, dir, point2_t{2.f, 1.f}), cos_inc_angle,
                tol);

    // Coordinate transformations
    point3_t glob_pos = {4.f, 7.f, 4.f};
    point3_t local = disc.global_to_local(ctx, glob_pos, {});
    point2_t bound = disc.global_to_bound(ctx, glob_pos, {});

    ASSERT_NEAR(local[0], std::sqrt(65.f), tol);
    ASSERT_NEAR(local[1], std::atan2(7.f, 4.f), tol);
    ASSERT_NEAR(bound[0], local[0], tol);
    ASSERT_NEAR(bound[1], local[1], tol);

    // Roundtrip
    point3_t global = disc.local_to_global(ctx, local, {});
    point3_t global2 = disc.bound_to_global(ctx, bound, {});

    ASSERT_NEAR(glob_pos[0], global[0], tol);
    ASSERT_NEAR(glob_pos[1], global[1], tol);
    ASSERT_NEAR(glob_pos[2], global[2], tol);

    ASSERT_NEAR(global2[0], glob_pos[0], tol);
    ASSERT_NEAR(global2[1], glob_pos[1], tol);
    // The bound transform assumes the point is on surface
    ASSERT_NEAR(global2[2], disc_translation[2], tol);

    // Test the material
    ASSERT_TRUE(disc.has_material());
    const auto* mat_param = disc.material_parameters({0.f, 0.f});
    ASSERT_TRUE(mat_param);
    ASSERT_EQ(*mat_param, toy_cfg.mapped_material());

    //
    // Rectangle
    //
    const auto rec_descr = toy_det.surfaces()[604u];
    const auto rec = tracking_surface{toy_det, rec_descr};

    // IDs
    ASSERT_EQ(rec.barcode(), rec_descr.barcode());
    ASSERT_EQ(rec.volume(), 9u);
    ASSERT_EQ(rec.index(), 604u);
    ASSERT_EQ(rec.id(), surface_id::e_sensitive);
    ASSERT_EQ(rec.shape_id(), detector_t::masks::id::e_rectangle2);
    ASSERT_TRUE(rec.is_sensitive());
    ASSERT_FALSE(rec.is_passive());
    ASSERT_FALSE(rec.is_portal());

    // Transformation matrix
    const auto& rec_translation = rec.transform(ctx).translation();
    ASSERT_NEAR(rec_translation[0], -71.902099f, tol);
    ASSERT_NEAR(rec_translation[1], -7.081735f, tol);
    ASSERT_NEAR(rec_translation[2], -455.f, tol);
    center = rec.center(ctx);
    ASSERT_NEAR(center[0], -71.902099f, tol);
    ASSERT_NEAR(center[1], -7.081735f, tol);
    ASSERT_NEAR(center[2], -455.f, tol);

    // Surface normal
    // trigger all code paths
    global = rec.transform(ctx).vector_to_global(z_axis);
    ASSERT_EQ(rec.normal(ctx, point3_t{0.f, 0.f, 0.f}), global);
    ASSERT_EQ(rec.normal(ctx, point2_t{0.f, 0.f}), global);

    // Incidence angle
    dir = vector::normalize(global);
    ASSERT_NEAR(rec.cos_angle(ctx, dir, point3_t{0.f, 0.f, 0.f}), 1.f, tol);
    ASSERT_NEAR(rec.cos_angle(ctx, dir, point2_t{0.f, 0.f}), 1.f, tol);

    dir = vector::normalize(vector3_t{0.f, -global[2], global[1]});
    ASSERT_NEAR(rec.cos_angle(ctx, dir, point3_t{1.f, 0.5f, 0.f}), 0.f, tol);
    ASSERT_NEAR(rec.cos_angle(ctx, dir, point2_t{1.f, 0.5f}), 0.f, tol);

    dir = vector::normalize(vector3_t{1.f, 0.f, constant<scalar_t>::pi});
    cos_inc_angle = std::fabs(vector::dot(dir, global));
    ASSERT_NEAR(rec.cos_angle(ctx, dir, point3_t{2.f, 1.f, 0.f}), cos_inc_angle,
                tol);
    ASSERT_NEAR(rec.cos_angle(ctx, dir, point2_t{2.f, 1.f}), cos_inc_angle,
                tol);

    // Coordinate transformation roundtrip
    glob_pos = {4.f, 7.f, 4.f};

    local = rec.global_to_local(ctx, glob_pos, {});
    global = rec.local_to_global(ctx, local, {});
    ASSERT_NEAR(glob_pos[0], global[0], tol);
    ASSERT_NEAR(glob_pos[1], global[1], tol);
    ASSERT_NEAR(glob_pos[2], global[2], tol);

    glob_pos = {-71.902099f, -7.081735f, -460.f};

    local = rec.global_to_local(ctx, glob_pos, {});
    global = rec.local_to_global(ctx, local, {});
    ASSERT_NEAR(glob_pos[0], global[0], tol);
    ASSERT_NEAR(glob_pos[1], global[1], tol);
    ASSERT_NEAR(glob_pos[2], global[2], tol);

    bound = rec.global_to_bound(ctx, glob_pos, {});
    global = rec.bound_to_global(ctx, bound, {});
    ASSERT_NEAR(global[0], glob_pos[0], tol);
    ASSERT_NEAR(global[1], glob_pos[1], tol);
    ASSERT_NEAR(global[2], glob_pos[2], tol);

    // Test the material (no material on sensitive surfaces)
    ASSERT_FALSE(rec.has_material());
    mat_param = rec.material_parameters({0.f, 0.f});
    ASSERT_FALSE(mat_param);

    //
    // Concentric Cylinder
    //
    const auto cyl_descr = toy_det.surfaces()[8u];
    const auto cyl = tracking_surface{toy_det, cyl_descr};

    // IDs
    ASSERT_EQ(cyl.barcode(), cyl_descr.barcode());
    ASSERT_EQ(cyl.volume(), 0u);
    ASSERT_EQ(cyl.index(), 8u);
    ASSERT_EQ(cyl.id(), surface_id::e_portal);
    ASSERT_EQ(cyl.shape_id(), detector_t::masks::id::e_portal_cylinder2);
    ASSERT_FALSE(cyl.is_sensitive());
    ASSERT_FALSE(cyl.is_passive());
    ASSERT_TRUE(cyl.is_portal());

    // Transformation matrix
    const auto& cyl_translation = cyl.transform(ctx).translation();
    ASSERT_NEAR(cyl_translation[0], 0.f, tol);
    ASSERT_NEAR(cyl_translation[1], 0.f, tol);
    ASSERT_NEAR(cyl_translation[2], 0.f, tol);
    center = cyl.center(ctx);
    ASSERT_NEAR(center[0], 0.f, tol);
    ASSERT_NEAR(center[1], 0.f, tol);
    ASSERT_NEAR(center[2], 0.f, tol);

    // Surface normal
    // trigger all code paths
    constexpr scalar_t r{25.f * unit<scalar_t>::mm};
    const vector3_t x_axis{1.f, 0.f, 0.f};
    ASSERT_NEAR(getter::norm(cyl.normal(ctx, point3_t{0.f, 0.f, r}) - x_axis),
                0.f, tol);
    ASSERT_NEAR(getter::norm(cyl.normal(ctx, point2_t{0.f, 0.f}) - x_axis), 0.f,
                tol);
    ASSERT_NEAR(
        getter::norm(
            cyl.normal(ctx, point3_t{r * constant<scalar_t>::pi, 0.f, r}) +
            x_axis),
        0.f, tol);
    ASSERT_NEAR(getter::norm(
                    cyl.normal(ctx, point2_t{r * constant<scalar_t>::pi, 0.f}) +
                    x_axis),
                0.f, tol);

    const vector3_t y_axis{0.f, 1.f, 0.f};
    ASSERT_NEAR(
        getter::norm(
            cyl.normal(ctx, point3_t{r * constant<scalar_t>::pi_2, 0.f, r}) -
            y_axis),
        0.f, tol);
    ASSERT_NEAR(
        getter::norm(
            cyl.normal(ctx, point2_t{r * constant<scalar_t>::pi_2, 0.f}) -
            y_axis),
        0.f, tol);

    // Incidence angle
    ASSERT_NEAR(cyl.cos_angle(ctx, x_axis,
                              point3_t{r * constant<scalar_t>::pi, 0.f, r}),
                1.f, tol);
    ASSERT_NEAR(
        cyl.cos_angle(ctx, x_axis, point2_t{r * constant<scalar_t>::pi, 0.f}),
        1.f, tol);

    ASSERT_NEAR(cyl.cos_angle(ctx, z_axis,
                              point3_t{r * constant<scalar_t>::pi_2, 0.f, r}),
                0.f, tol);
    ASSERT_NEAR(
        cyl.cos_angle(ctx, z_axis, point2_t{r * constant<scalar_t>::pi_2, 0.f}),
        0.f, tol);

    dir = vector::normalize(vector3_t{1.f, 1.f, 0.f});
    ASSERT_NEAR(cyl.cos_angle(ctx, dir, point3_t{0.f, 1.f, r}),
                constant<scalar_t>::inv_sqrt2, tol);
    ASSERT_NEAR(cyl.cos_angle(ctx, dir, point2_t{0.f, 1.f}),
                constant<scalar_t>::inv_sqrt2, tol);

    // Coordinate transformation roundtrip
    glob_pos = {4.f, 7.f, 4.f};

    local = cyl.global_to_local(ctx, glob_pos, {});
    global = cyl.local_to_global(ctx, local, {});
    ASSERT_NEAR(glob_pos[0], global[0], tol);
    ASSERT_NEAR(glob_pos[1], global[1], tol);
    ASSERT_NEAR(glob_pos[2], global[2], tol);

    glob_pos = {constant<scalar_t>::inv_sqrt2 * r,
                constant<scalar_t>::inv_sqrt2 * r, 2.f};

    local = cyl.global_to_local(ctx, glob_pos, {});
    global = cyl.local_to_global(ctx, local, {});
    ASSERT_NEAR(glob_pos[0], global[0], tol);
    ASSERT_NEAR(glob_pos[1], global[1], tol);
    ASSERT_NEAR(glob_pos[2], global[2], tol);

    bound = cyl.global_to_bound(ctx, glob_pos, {});
    global = cyl.bound_to_global(ctx, bound, {});
    ASSERT_NEAR(global[0], glob_pos[0], tol);
    ASSERT_NEAR(global[1], glob_pos[1], tol);
    ASSERT_NEAR(global[2], glob_pos[2], tol);

    // Test the material
    ASSERT_TRUE(cyl.has_material());
    mat_param = cyl.material_parameters({0.f, 0.f});
    ASSERT_TRUE(mat_param);
    ASSERT_EQ(*mat_param, toy_cfg.mapped_material());
}

/// This tests the functionality of a tracking surface interface for a wire
/// chamber detector
GTEST_TEST(detray_geometry, surface_wire_chamber) {

    using namespace detray;

    using detector_t = detector<default_metadata>;

    using scalar_t = tracking_surface<detector_t>::scalar_type;
    using point2_t = tracking_surface<detector_t>::point2_type;
    using point3_t = tracking_surface<detector_t>::point3_type;
    using vector3_t = tracking_surface<detector_t>::vector3_type;

    vecmem::host_memory_resource host_mr;
    wire_chamber_config<> cfg{};
    const auto [wire_chmbr, names] = build_wire_chamber(host_mr, cfg);

    auto ctx = typename detector_t::geometry_context{};

    //
    // Line
    //
    const auto line_descr = wire_chmbr.surfaces()[23u];
    const auto line = tracking_surface{wire_chmbr, line_descr};

    // IDs
    ASSERT_EQ(line.barcode(), line_descr.barcode());
    ASSERT_EQ(line.volume(), 1u);
    ASSERT_EQ(line.index(), 23u);
    ASSERT_EQ(line.id(), surface_id::e_sensitive);
    ASSERT_EQ(line.shape_id(), detector_t::masks::id::e_drift_cell);
    ASSERT_TRUE(line.is_sensitive());
    ASSERT_FALSE(line.is_passive());
    ASSERT_FALSE(line.is_portal());

    // Transformation matrix
    const auto line_translation = line.transform(ctx).translation();
    ASSERT_NEAR(line_translation[0], 412.858582f, tol);
    ASSERT_NEAR(line_translation[1], 299.412414f, tol);
    ASSERT_NEAR(line_translation[2], 0.f, tol);
    auto center = line.center(ctx);
    ASSERT_NEAR(center[0], 412.858582f, tol);
    ASSERT_NEAR(center[1], 299.412414f, tol);
    ASSERT_NEAR(center[2], 0.f, tol);

    // Surface normal
    const auto z_axis = vector3_t{0.f, 0.f, 1.f};
    point3_t global = line.transform(ctx).vector_to_global(z_axis);
    // trigger all code paths
    ASSERT_EQ(line.normal(ctx, point3_t{1.f, 0.f, 0.f}), global);
    ASSERT_EQ(line.normal(ctx, point2_t{1.f, 0.f}), global);
    ASSERT_EQ(line.normal(ctx, point3_t{-1.f, 0.f, 0.f}), global);
    ASSERT_EQ(line.normal(ctx, point2_t{-1.f, 0.f}), global);

    // Cos incidence angle
    auto dir = vector::normalize(global);
    ASSERT_NEAR(line.cos_angle(ctx, dir, point3_t{1.f, 0.f, 0.f}), 1.f, tol);
    ASSERT_NEAR(line.cos_angle(ctx, dir, point2_t{1.f, 0.f}), 1.f, tol);

    dir = vector::normalize(vector3_t{0.f, -global[2], global[1]});
    ASSERT_NEAR(line.cos_angle(ctx, dir, point3_t{1.f, 100.f, 0.f}), 0.f, tol);
    ASSERT_NEAR(line.cos_angle(ctx, dir, point2_t{1.f, 100.f}), 0.f, tol);

    dir = vector3_t{-0.685475f, -0.0404595f, 0.726971f};
    ASSERT_NEAR(line.cos_angle(ctx, dir, point3_t{2.f, 1.f, 0.f}),
                constant<scalar_t>::inv_sqrt2, 0.0005);
    ASSERT_NEAR(line.cos_angle(ctx, dir, point2_t{2.f, 1.f}),
                constant<scalar_t>::inv_sqrt2, 0.0005);

    // Coordinate transformation roundtrip
    point3_t glob_pos = {4.f, 7.f, 4.f};

    point3_t local = line.global_to_local(ctx, glob_pos, dir);
    global = line.local_to_global(ctx, local, dir);

    // @TODO: Needs a reduced tolerance, why?
    scalar_t red_tol{0.00015f};
    ASSERT_NEAR(glob_pos[0], global[0], red_tol);
    ASSERT_NEAR(glob_pos[1], global[1], red_tol);
    ASSERT_NEAR(glob_pos[2], global[2], red_tol);

    glob_pos = center;

    local = line.global_to_local(ctx, glob_pos, dir);
    global = line.local_to_global(ctx, local, dir);
    red_tol = 7.f * 1e-5f;
    ASSERT_NEAR(glob_pos[0], global[0], red_tol);
    ASSERT_NEAR(glob_pos[1], global[1], red_tol);
    ASSERT_NEAR(glob_pos[2], global[2], red_tol);

    point2_t bound = line.global_to_bound(ctx, glob_pos, dir);
    global = line.bound_to_global(ctx, bound, dir);
    ASSERT_NEAR(global[0], glob_pos[0], red_tol);
    ASSERT_NEAR(global[1], glob_pos[1], red_tol);
    ASSERT_NEAR(global[2], glob_pos[2], red_tol);

    // Test the material
    ASSERT_TRUE(line.has_material());
    const auto* mat_param = line.material_parameters({0.f, 0.f});
    ASSERT_TRUE(mat_param);
    ASSERT_EQ(*mat_param, tungsten<scalar_t>());
}
