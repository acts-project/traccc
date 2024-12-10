/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2020-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/containers.hpp"
#include "detray/definitions/detail/indexing.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/geometry/coordinates/cartesian2D.hpp"

// System include(s)
#include <limits>
#include <ostream>
#include <string_view>

namespace detray {

/// @brief Geometrical shape of a rectangle2D.
///
/// It is defined by half length in local0 coordinates bounds[0] and bounds[1]
class rectangle2D {
    public:
    /// The name for this shape
    static constexpr std::string_view name = "rectangle2D";

    enum boundaries : unsigned int {
        e_half_x = 0u,
        e_half_y = 1u,
        e_size = 2u,
    };

    /// Container definition for the shape boundary values
    template <typename scalar_t>
    using bounds_type = darray<scalar_t, boundaries::e_size>;

    /// Local coordinate frame for boundary checks
    template <typename algebra_t>
    using local_frame_type = cartesian2D<algebra_t>;

    /// Dimension of the local coordinate system
    static constexpr std::size_t dim{2u};

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
        const bounds_type<scalar_t> &bounds, const point_t &loc_p) const {

        return math::min(math::fabs(math::fabs(loc_p[0]) - bounds[e_half_x]),
                         math::fabs(math::fabs(loc_p[1]) - bounds[e_half_y]));
    }

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
        return (math::fabs(loc_p[0]) <= (bounds[e_half_x] + tol) &&
                math::fabs(loc_p[1]) <= (bounds[e_half_y] + tol));
    }

    /// @brief Measure of the shape: Area
    ///
    /// @param bounds the boundary values for this shape
    ///
    /// @returns the rectangle area on the plane
    template <typename scalar_t>
    DETRAY_HOST_DEVICE constexpr scalar_t measure(
        const bounds_type<scalar_t> &bounds) const {
        return area(bounds);
    }

    /// @brief The area of a the shape
    ///
    /// @param bounds the boundary values for this shape
    ///
    /// @returns the rectangle area.
    template <typename scalar_t>
    DETRAY_HOST_DEVICE constexpr scalar_t area(
        const bounds_type<scalar_t> &bounds) const {
        return 4.f * bounds[e_half_x] * bounds[e_half_y];
    }

    /// @brief Lower and upper point for minimal axis aligned bounding box.
    ///
    /// Computes the min and max vertices in a local cartesian frame.
    ///
    /// @param bounds the boundary values for this shape
    /// @param env dynamic envelope around the shape
    ///
    /// @returns an array of coordinates that contains the lower point (first
    /// three values) and the upper point (latter three values) .
    template <typename algebra_t>
    DETRAY_HOST_DEVICE inline darray<dscalar<algebra_t>, 6> local_min_bounds(
        const bounds_type<dscalar<algebra_t>> &bounds,
        const dscalar<algebra_t> env =
            std::numeric_limits<dscalar<algebra_t>>::epsilon()) const {
        using scalar_t = dscalar<algebra_t>;

        assert(env > 0.f);
        const scalar_t x_bound{bounds[e_half_x] + env};
        const scalar_t y_bound{bounds[e_half_y] + env};
        return {-x_bound, -y_bound, -env, x_bound, y_bound, env};
    }

    /// @returns the shapes centroid in local cartesian coordinates
    template <typename algebra_t>
    DETRAY_HOST_DEVICE dpoint3D<algebra_t> centroid(
        const bounds_type<dscalar<algebra_t>> &) const {

        return {0.f, 0.f, 0.f};
    }

    /// Generate vertices in local cartesian frame
    ///
    /// @param bounds the boundary values for the stereo annulus
    /// @param n_seg is the number of line segments
    ///
    /// @return a generated list of vertices
    template <typename algebra_t>
    DETRAY_HOST dvector<dpoint3D<algebra_t>> vertices(
        const bounds_type<dscalar<algebra_t>> &bounds,
        dindex /*ignored*/) const {

        using point3_t = dpoint3D<algebra_t>;

        // left hand lower corner
        point3_t lh_lc{-bounds[e_half_x], -bounds[e_half_y], 0.f};
        // right hand lower corner
        point3_t rh_lc{bounds[e_half_x], -bounds[e_half_y], 0.f};
        // right hand upper corner
        point3_t rh_uc{bounds[e_half_x], bounds[e_half_y], 0.f};
        // left hand upper corner
        point3_t lh_uc{-bounds[e_half_x], bounds[e_half_y], 0.f};

        // Return the confining vertices
        return {lh_lc, rh_lc, rh_uc, lh_uc};
    }

    /// @brief Check consistency of boundary values.
    ///
    /// @param bounds the boundary values for this shape
    /// @param os output stream for error messages
    ///
    /// @return true if the bounds are consistent.
    template <typename scalar_t>
    DETRAY_HOST constexpr bool check_consistency(
        const bounds_type<scalar_t> &bounds, std::ostream &os) const {

        if (constexpr auto tol{10.f * std::numeric_limits<scalar_t>::epsilon()};
            bounds[e_half_x] < tol || bounds[e_half_y] < tol) {
            os << "ERROR: Half lengths must be in the range (0, numeric_max)"
               << std::endl;
            return false;
        }

        return true;
    }
};

}  // namespace detray
