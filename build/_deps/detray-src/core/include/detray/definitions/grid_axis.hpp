/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace detray::axis {

/// axis bounds names.
///
/// open: adds additional over-/undeflow bins.
/// closed: over-/underflow values are mapped into the bin range.
/// circular: over-/underflow values wrap around.
enum class bounds {
    e_open = 0,
    e_closed = 1,
    e_circular = 2,
};

/// axis coordinate names. Used to get a specific axes from an axes collection.
///
/// x, y, z: cartesian coordinate axes.
/// r, phi: polar coordinate axes.
/// rphi, cyl_z: 2D cylinder axes (3D cylinder uses r, phi, z).
enum class label {
    e_x = 0,
    e_y = 1,
    e_z = 2,
    e_r = 0,
    e_phi = 1,
    e_rphi = 0,
    e_cyl_z = 1,
};

/// axis binning type names.
///
/// regular: same sized bins along the axis.
/// irregular: every bin can have a different size along the axis.
enum class binning {
    e_regular = 0,
    e_irregular = 1,
};

}  // namespace detray::axis
