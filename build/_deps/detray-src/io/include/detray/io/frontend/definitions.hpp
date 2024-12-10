/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace detray {

using real_io = double;

/// The following enums are defined per detector in the detector metadata
namespace io {

enum class format { json = 0u };

/// Enumerate the shape primitives globally
enum class shape_id : unsigned int {
    annulus2 = 0u,
    cuboid3 = 1u,
    cylinder2 = 2u,
    cylinder3 = 3u,
    portal_cylinder2 = 4u,
    rectangle2 = 5u,
    ring2 = 6u,
    trapezoid2 = 7u,
    drift_cell = 8u,
    straw_tube = 9u,
    single1 = 10u,
    single2 = 11u,
    single3 = 12u,
    n_shapes = 13u,
    unknown = n_shapes
};

/// Enumerate the different material types
enum class material_id : unsigned int {
    // Material texture (grid) shapes
    annulus2_map = 0u,
    rectangle2_map = 1u,
    cuboid3_map = 2u,
    concentric_cylinder2_map = 3u,
    cylinder2_map = 4u,
    cylinder3_map = 5u,
    ring2_map = 0u,
    trapezoid2_map = 1u,
    // Homogeneous materials
    slab = 6u,
    rod = 7u,
    raw_material = 8u,
    n_mats = 9u,
    unknown = n_mats
};

/// Enumerate the different acceleration data structures
enum class accel_id : unsigned int {
    brute_force = 0u,                // try all
    cartesian2_grid = 1u,            // rectangle, trapezoid, (triangle) grids
    cuboid3_grid = 2u,               // cuboid grid
    polar2_grid = 3u,                // ring/disc, annulus grids
    concentric_cylinder2_grid = 4u,  // 2D concentric cylinder grid
    cylinder2_grid = 5u,             // 2D cylinder grid
    cylinder3_grid = 6u,             // 3D cylinder grid
    n_accel = 7u,
    unknown = n_accel
};

}  // namespace io

}  // namespace detray
