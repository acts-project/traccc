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
#include "detray/definitions/detail/math.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/definitions/units.hpp"
#include "detray/geometry/coordinates/cylindrical2D.hpp"

// System include(s)
#include <limits>
#include <ostream>
#include <string_view>

namespace detray {

/// @brief Geometrical shape of a 2D cylinder.
///
/// It is defined by r and the two half lengths rel to the coordinate center.
class cylinder2D {
    public:
    /// The name for this shape
    static constexpr std::string_view name = "cylinder2D";

    enum boundaries : unsigned int {
        e_r = 0u,
        e_lower_z = 1u,
        e_upper_z = 2u,
        e_size = 3u,
    };

    /// Container definition for the shape boundary values
    template <typename scalar_t>
    using bounds_type = darray<scalar_t, boundaries::e_size>;

    /// Local coordinate frame for boundary checks
    template <typename algebra_t>
    using local_frame_type = cylindrical2D<algebra_t>;

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

        return math::min(math::fabs(loc_p[1] - bounds[e_lower_z]),
                         math::fabs(bounds[e_upper_z] - loc_p[1]));
    }

    /// @brief Check boundary values for a local point.
    ///
    /// @note the point is expected to be given in local coordinates by the
    /// caller. For the conversion from global cartesian coordinates, the
    /// nested @c shape struct can be used. The point is assumed to be in
    /// the cylinder 2D frame (r * phi, z).
    ///
    /// @tparam is_rad_check whether the radial bound should be checked in this
    /// call.
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

        return ((bounds[e_lower_z] - tol) <= loc_p[1] &&
                loc_p[1] <= (bounds[e_upper_z] + tol));
    }

    /// @brief Measure of the shape: Area
    ///
    /// @param bounds the boundary values for this shape
    ///
    /// @returns the cylinder area on the cylinder of radius r.
    template <typename scalar_t>
    DETRAY_HOST_DEVICE constexpr scalar_t measure(
        const bounds_type<scalar_t> &bounds) const {
        return area(bounds);
    }

    /// @brief The area of a the shape
    ///
    /// @param bounds the boundary values for this shape
    ///
    /// @returns the stereo annulus area.
    template <typename scalar_t>
    DETRAY_HOST_DEVICE constexpr scalar_t area(
        const bounds_type<scalar_t> &bounds) const {
        return 2.f * constant<scalar_t>::pi * bounds[e_r] *
               (bounds[e_upper_z] - bounds[e_lower_z]);
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
        assert(env > 0.f);

        const dscalar<algebra_t> xy_bound{bounds[e_r] + env};
        return {-xy_bound, -xy_bound, bounds[e_lower_z] - env,
                xy_bound,  xy_bound,  bounds[e_upper_z] + env};
    }

    /// @returns the shapes centroid in local cartesian coordinates
    template <typename algebra_t>
    DETRAY_HOST_DEVICE dpoint3D<algebra_t> centroid(
        const bounds_type<dscalar<algebra_t>> &bounds) const {

        return {0.f, 0.f, 0.5f * (bounds[e_lower_z] + bounds[e_upper_z])};
    }

    /// Generate vertices in local cartesian frame
    ///
    /// @param bounds the boundary values for the cylinder
    /// @param n_seg is the number of line segments
    ///
    /// @return a generated list of vertices
    template <typename algebra_t>
    DETRAY_HOST dvector<dpoint3D<algebra_t>> vertices(
        const bounds_type<dscalar<algebra_t>> &, dindex) const {
        throw std::runtime_error(
            "Vertex generation for cylinders is not implemented");
        return {};
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

        constexpr auto tol{10.f * std::numeric_limits<scalar_t>::epsilon()};

        if (bounds[e_r] < tol) {
            os << "ERROR: Radius must be in the range (0, numeric_max)"
               << std::endl;
            return false;
        }
        if (bounds[e_lower_z] >= bounds[e_upper_z] ||
            math::fabs(bounds[e_lower_z] - bounds[e_upper_z]) < tol) {
            os << "ERROR: Neg. half length must be smaller than pos. half "
                  "length.";
            return false;
        }

        return true;
    }
};

}  // namespace detray
