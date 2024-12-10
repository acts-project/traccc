/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "detray/definitions/detail/algebra.hpp"
#include "detray/definitions/detail/qualifiers.hpp"

// System include(s)
#include <array>

namespace detray {

template <typename vector3_t>
struct unit_vectors {

    /// Construct the first curvilinear unit vector `U` for the given direction.
    ///
    /// @param dir is the input direction vector
    /// @returns a normalized vector in the x-y plane orthogonal to the
    /// direction.
    ///
    /// The special case of the direction vector pointing along the z-axis is
    /// handled by forcing the unit vector to along the x-axis.
    DETRAY_HOST_DEVICE inline vector3_t make_curvilinear_unit_u(
        const vector3_t& dir) {

        vector3_t unit_u{0.f, 0.f, 0.f};
        // explicit version of U = Z x T
        unit_u[0] = -dir[1];
        unit_u[1] = dir[0];

        const auto scale = getter::norm(unit_u);

        // if the absolute scale is tiny, the initial direction vector is
        // aligned with the z-axis. the ZxT product is ill-defined since any
        // vector in the x-y plane would be orthogonal to the direction. fix the
        // U unit vector along the x-axis to avoid this numerical instability.
        if (scale < 1e-6f) {
            unit_u[0] = 1;
            unit_u[1] = 0;
        } else {
            unit_u = unit_u * (1.f / scale);
        }

        return unit_u;
    }

    /// Construct the curvilinear unit vectors `U` and `V` for the given
    /// direction.
    ///
    /// @param dir is the input direction vector
    /// @returns normalized unit vectors `U` and `V` orthogonal to the
    /// direction.
    ///
    /// With `T` the normalized input direction, the three vectors `U`, `V`, and
    /// `T` form an orthonormal basis set, i.e. they satisfy
    ///
    ///     U x V = T
    ///     V x T = U
    ///     T x U = V
    ///
    /// with the additional condition that `U` is located in the global x-y
    /// plane.
    DETRAY_HOST_DEVICE inline std::array<vector3_t, 2>
    make_curvilinear_unit_vectors(const vector3_t& dir) {

        std::array<vector3_t, 2> uv;
        uv[0] = make_curvilinear_unit_u(dir);
        uv[1] = vector::normalize(vector::cross(dir, uv[0]));

        return uv;
    }
};

}  // namespace detray
