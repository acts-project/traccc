/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace detray::detail {

/// Generate phi tolerance from distance tolerance
///
/// @param tol is the distance tolerance in mm
/// @param radius is the radius of the shape
///
/// @return the opening angle of a chord the size of tol (= 2*arcsin(c/(2r)))
/// using a small angle approximation
template <typename scalar_t>
constexpr scalar_t phi_tolerance(scalar_t tol, scalar_t radius) {
    return tol / radius;
}

}  // namespace detray::detail
