/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/geometry/coordinates/cartesian2D.hpp"

// System include(s)
#include <limits>
#include <string_view>

namespace detray::tutorial {

/// @brief Geometrical shape of a square.
///
/// It is defined by the half length of the square (bounds[0]),
/// and can be checked with a tolerance in t.
class square2D {
    public:
    /// The name for this shape
    static constexpr std::string_view name = "square2D";

    enum boundaries : unsigned int {
        e_half_length = 0,  // < boundary value: the half length of the square
        e_size = 1u,        // < Number of boundary values for this shape
    };

    /// Container definition for the shape boundary values
    template <typename scalar_t>
    using bounds_type = darray<scalar_t, boundaries::e_size>;

    /// Local coordinate frame for boundary checks: cartesian
    template <typename algebra_t>
    using local_frame_type = cartesian2D<algebra_t>;

    /// Dimension of the local coordinate system
    static constexpr std::size_t dim{2u};

    /// @brief Check boundary values for a local point.
    ///
    /// @note the point is expected to be given in local coordinates by the
    /// caller. For the conversion from global cartesian coordinates, the
    /// nested @c shape struct can be used.
    ///
    /// @param bounds the boundary values for this shape
    /// @param loc_p the point to be checked in the local coordinate system
    /// @param tol dynamic tolerance determined by caller
    ///
    /// @return true if the local point lies within the given boundaries.
    template <typename scalar_t, typename point_t>
    DETRAY_HOST_DEVICE inline auto check_boundaries(
        const bounds_type<scalar_t> &bounds, const point_t &loc_p,
        const scalar_t tol = std::numeric_limits<scalar_t>::epsilon()) const {
        return (math::fabs(loc_p[0]) <= bounds[e_half_length] + tol &&
                math::fabs(loc_p[1]) <= bounds[e_half_length] + tol);
    }

    /// @brief Lower and upper point for minimal axis aligned bounding box.
    ///
    /// Computes the min and max vertices in a local cartesian frame.
    ///
    /// @param bounds the boundary values for this shape
    /// @param env dynamic envelope around the shape
    ///
    /// @returns and array of coordinates that contains the lower point (first
    /// three values) and the upper point (latter three values) .
    template <typename algebra_t>
    DETRAY_HOST_DEVICE inline std::array<dscalar<algebra_t>, 6>
    local_min_bounds(
        const bounds_type<dscalar<algebra_t>> &bounds,
        const dscalar<algebra_t> env =
            std::numeric_limits<dscalar<algebra_t>>::epsilon()) const {
        assert(env > 0.f);
        const dscalar<algebra_t> bound{bounds[e_half_length] + env};
        return {-bound, -bound, -env, bound, bound, env};
    }
};

}  // namespace detray::tutorial
