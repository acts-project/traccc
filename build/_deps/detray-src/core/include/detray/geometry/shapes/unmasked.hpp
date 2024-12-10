/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2020-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/algebra.hpp"
#include "detray/definitions/detail/containers.hpp"
#include "detray/definitions/detail/indexing.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/geometry/coordinates/cartesian2D.hpp"

// System include(s)
#include <limits>
#include <ostream>
#include <string_view>

namespace detray {

/// @brief Generic shape without boundaries.
template <std::size_t DIM = 2>
class unmasked {
    public:
    /// The name for this shape
    static constexpr std::string_view name = "unmasked";

    enum boundaries : unsigned int { e_size = 1u };

    /// Container definition for the shape boundary values
    template <typename scalar_t>
    using bounds_type = darray<scalar_t, boundaries::e_size>;

    /// Local coordinate frame for boundary checks
    template <typename algebra_t>
    using local_frame_type = cartesian2D<algebra_t>;

    /// Dimension of the local coordinate system
    static constexpr std::size_t dim{DIM};

    /// @brief Find the minimum distance to any boundary.
    ///
    /// @note the point is expected to be given in local coordinates by the
    /// caller.
    ///
    /// @param bounds the boundary values for this shape
    /// @param loc_p the point to be checked in the local coordinate system
    ///
    /// @return the minimum distance.
    template <typename scalar_t, typename point_t>
    DETRAY_HOST_DEVICE inline scalar_t min_dist_to_boundary(
        const bounds_type<scalar_t>& /*bounds*/,
        const point_t& /*loc_p*/) const {
        return std::numeric_limits<scalar_t>::max();
    }

    /// @brief Check boundary values for a local point.
    ///
    /// @tparam bounds_t any type of boundary values
    ///
    /// @note the parameters are ignored
    ///
    /// @return always true
    template <typename scalar_t, typename point_t>
    DETRAY_HOST_DEVICE inline auto check_boundaries(
        const bounds_type<scalar_t>& /*bounds*/, const point_t& /*loc_p*/,
        const scalar_t /*tol*/) const {
        return true;
    }

    /// @brief Measure of the shape: Inf
    ///
    /// @param bounds the boundary values for this shape
    ///
    /// @returns Inf.
    template <typename scalar_t>
    DETRAY_HOST_DEVICE constexpr scalar_t measure(
        const bounds_type<scalar_t>& bounds) const {
        if constexpr (dim == 2) {
            return area(bounds);
        } else {
            return volume(bounds);
        }
    }

    /// @brief The area of a the shape
    ///
    /// @param bounds the boundary values for this shape
    ///
    /// @returns the stereo annulus area.
    template <typename scalar_t>
    DETRAY_HOST_DEVICE constexpr scalar_t area(
        const bounds_type<scalar_t>&) const {
        return std::numeric_limits<scalar_t>::max();
    }

    /// @brief The volume of a the shape
    ///
    /// @param bounds the boundary values for this shape
    ///
    /// @returns Inf.
    template <typename scalar_t>
    DETRAY_HOST_DEVICE constexpr scalar_t volume(
        const bounds_type<scalar_t>&) const {
        return std::numeric_limits<scalar_t>::max();
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
    DETRAY_HOST_DEVICE inline darray<dscalar<algebra_t>, 6> local_min_bounds(
        const bounds_type<dscalar<algebra_t>>& /*bounds*/,
        const dscalar<algebra_t> /*env*/ =
            std::numeric_limits<dscalar<algebra_t>>::epsilon()) const {

        using scalar_t = dscalar<algebra_t>;
        constexpr scalar_t inv{detail::invalid_value<scalar_t>()};

        return {-inv, -inv, -inv, inv, inv, inv};
    }

    /// @returns the shapes centroid in global cartesian coordinates
    template <typename algebra_t>
    DETRAY_HOST_DEVICE dpoint3D<algebra_t> centroid(
        const bounds_type<dscalar<algebra_t>>&) const {
        return {0.f, 0.f, 0.f};
    }

    /// Generate vertices in local cartesian frame
    ///
    /// @param bounds the boundary values
    /// @param n_seg is the number of line segments
    ///
    /// @return a generated list of vertices
    template <typename algebra_t>
    DETRAY_HOST dvector<dpoint3D<algebra_t>> vertices(
        const bounds_type<dscalar<algebra_t>>& bounds, dindex) const {
        return local_min_bounds(bounds);
    }

    /// @brief Check consistency of boundary values.
    ///
    /// @param bounds the boundary values for this shape
    /// @param os output stream for error messages
    ///
    /// @return true if the bounds are consistent.
    template <typename scalar_t>
    DETRAY_HOST constexpr bool check_consistency(
        const bounds_type<scalar_t>& /*bounds*/,
        const std::ostream& /*os*/) const {
        return true;
    }
};

}  // namespace detray
