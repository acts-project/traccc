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
#include "detray/geometry/coordinates/cylindrical3D.hpp"
#include "detray/geometry/detail/shape_utils.hpp"

// System include(s)
#include <limits>
#include <ostream>
#include <string_view>

namespace detray {

/// @brief Geometrical shape of a full 3D cylinder.
///
/// It is defined by r and the two half lengths rel to the coordinate center.
class cylinder3D {
    public:
    /// The name for this shape
    static constexpr std::string_view name = "cylinder3D";

    enum boundaries : unsigned int {
        e_min_r = 0u,
        e_min_phi = 1u,
        e_min_z = 2u,
        e_max_r = 3u,
        e_max_phi = 4u,
        e_max_z = 5u,
        e_size = 6u,
    };

    /// Container definition for the shape boundary values
    template <typename scalar_t>
    using bounds_type = darray<scalar_t, boundaries::e_size>;

    /// Local coordinate frame for boundary checks
    template <typename algebra_t>
    using local_frame_type = cylindrical3D<algebra_t>;

    /// Dimension of the local coordinate system
    static constexpr std::size_t dim{3u};

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

        const scalar_t min_r_dist =
            math::min(math::fabs(loc_p[0] - bounds[e_min_r]),
                      math::fabs(bounds[e_max_r] - loc_p[0]));
        const scalar_t min_phi_dist =
            math::min(math::fabs(loc_p[1] - bounds[e_min_phi]),
                      math::fabs(bounds[e_max_phi] - loc_p[1]));
        const scalar_t min_z_dist =
            math::min(math::fabs(loc_p[2] - bounds[e_min_z]),
                      math::fabs(bounds[e_max_z] - loc_p[2]));

        // Use the chord for the phi distance
        return math::min(
            math::min(min_r_dist,
                      2.f * loc_p[0] * math::sin(0.5f * min_phi_dist)),
            min_z_dist);
    }

    /// @brief Check boundary values for a local point.
    ///
    /// @note the point is expected to be given in local coordinates by the
    /// caller. For the conversion from global cartesian coordinates, the
    /// nested @c shape struct can be used. The point is assumed to be in
    /// the cylinder 3D frame.
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

        const scalar_t phi_tol = detail::phi_tolerance(tol, loc_p[0]);

        return ((bounds[e_min_r] - tol) <= loc_p[0] &&
                (bounds[e_min_phi] - phi_tol) <= loc_p[1] &&
                (bounds[e_min_z] - tol) <= loc_p[2] &&
                loc_p[0] <= (bounds[e_max_r] + tol) &&
                loc_p[1] <= (bounds[e_max_phi] + phi_tol) &&
                loc_p[2] <= (bounds[e_max_z] + tol));
    }

    /// @brief Measure of the shape: Volume
    ///
    /// @param bounds the boundary values for this shape
    ///
    /// @returns the cylinder volume as parto of global space.
    template <typename scalar_t>
    DETRAY_HOST_DEVICE constexpr scalar_t measure(
        const bounds_type<scalar_t> &bounds) const {
        return volume(bounds);
    }

    /// @brief The volume of a the shape
    ///
    /// @param bounds the boundary values for this shape
    ///
    /// @returns the cylinder volume.
    template <typename scalar_t>
    DETRAY_HOST_DEVICE constexpr scalar_t volume(
        const bounds_type<scalar_t> &bounds) const {
        return constant<scalar>::pi * (bounds[e_max_z] - bounds[e_min_z]) *
               (bounds[e_max_r] * bounds[e_max_r] -
                bounds[e_min_r] * bounds[e_min_r]);
    }

    /// @brief Lower and upper point for minimal axis aligned bounding box.
    ///
    /// Computes the min and max vertices in a local cartesian frame.
    ///
    /// @param bounds the boundary values for this shape
    /// @param env dynamic envelope around the shape
    ///
    /// @returns and array of coordinates that contains the lower point (first
    /// three values) and the upper point (latter three values).
    // @todo: Look at phi - range for a better fit
    template <typename algebra_t>
    DETRAY_HOST_DEVICE inline darray<dscalar<algebra_t>, 6> local_min_bounds(
        const bounds_type<dscalar<algebra_t>> &bounds,
        const dscalar<algebra_t> env =
            std::numeric_limits<dscalar<algebra_t>>::epsilon()) const {

        assert(env > 0.f);
        const dscalar<algebra_t> r_bound{bounds[e_max_r] + env};
        return {-r_bound, -r_bound, bounds[e_min_z] - env,
                r_bound,  r_bound,  bounds[e_max_z] + env};
    }

    /// @returns the shapes centroid in local cartesian coordinates
    template <typename algebra_t>
    DETRAY_HOST_DEVICE dpoint3D<algebra_t> centroid(
        const bounds_type<dscalar<algebra_t>> &bounds) const {

        return 0.5f *
               dpoint3D<algebra_t>{0.f, (bounds[e_min_phi] + bounds[e_max_phi]),
                                   (bounds[e_min_z] + bounds[e_max_z])};
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
            "Vertex generation for 3D cylinders is not implemented");
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

        if (bounds[e_min_r] < tol) {
            os << "ERROR: Radii must be in the range (0, numeric_max)"
               << std::endl;
            return false;
        }
        if (bounds[e_min_r] >= bounds[e_max_r] ||
            math::fabs(bounds[e_min_r] - bounds[e_max_r]) < tol) {
            os << "ERROR: Min Radius must be smaller than max Radius.";
            return false;
        }
        if (bounds[e_min_z] >= bounds[e_max_z] ||
            math::fabs(bounds[e_min_z] - bounds[e_max_z]) < tol) {
            os << "ERROR: Min z must be smaller than max z.";
            return false;
        }

        return true;
    }
};

}  // namespace detray
