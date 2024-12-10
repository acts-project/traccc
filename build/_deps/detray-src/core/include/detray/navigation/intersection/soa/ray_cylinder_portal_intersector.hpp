/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/boolean.hpp"
#include "detray/definitions/detail/math.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/geometry/coordinates/concentric_cylindrical2D.hpp"
#include "detray/navigation/detail/ray.hpp"
#include "detray/navigation/intersection/intersection.hpp"
#include "detray/navigation/intersection/soa/ray_cylinder_intersector.hpp"

// System include(s)
#include <type_traits>

namespace detray {

/// @brief A functor to find intersections between a straight line and a
/// cylindrical portal surface.
///
/// With the way the navigation works, only the closest one of the two possible
/// intersection points is needed in the case of a cylinderical portal surface.
template <concepts::soa_algebra algebra_t, bool do_debug>
struct ray_intersector_impl<concentric_cylindrical2D<algebra_t>, algebra_t,
                            do_debug>
    : public ray_intersector_impl<cylindrical2D<algebra_t>, algebra_t,
                                  do_debug> {

    /// Linear algebra types
    /// @{
    using scalar_type = dscalar<algebra_t>;
    using point3_type = dpoint3D<algebra_t>;
    using vector3_type = dvector3D<algebra_t>;
    using transform3_type = dtransform3D<algebra_t>;
    /// @}

    template <typename surface_descr_t>
    using intersection_type =
        intersection2D<surface_descr_t, algebra_t, do_debug>;

    /// Operator function to find intersections between ray and cylinder mask
    ///
    /// @tparam mask_t is the input mask type
    /// @tparam surface_t is the type of surface handle
    ///
    /// @param ray is the input ray trajectory
    /// @param sf the surface handle the mask is associated with
    /// @param mask is the input mask that defines the surface extent
    /// @param trf is the surface placement transform
    /// @param mask_tolerance is the tolerance for mask edges
    ///
    /// @return the closest intersection
    template <typename surface_descr_t, typename mask_t,
              typename other_algebra_t>
    DETRAY_HOST_DEVICE inline intersection_type<surface_descr_t> operator()(
        const detail::ray<other_algebra_t> &ray, const surface_descr_t &sf,
        const mask_t &mask, const transform3_type &trf,
        const std::array<scalar_type, 2u> &mask_tolerance = {0.f, 1.f},
        const scalar_type mask_tol_scalor = 0.f,
        const scalar_type overstep_tol = 0.f) const {

        intersection_type<surface_descr_t> is;

        // Intersecting the cylinder from the inside yield one intersection
        // along the direction of the track and one behind it
        const auto qe = this->solve_intersection(ray, mask, trf);

        // None of the cylinders has a valid intersection
        if (detray::detail::all_of(qe.solutions() <= 0) ||
            detray::detail::all_of(qe.larger() <= overstep_tol)) {
            is.status = decltype(is.status)(false);
            return is;
        }

        // Only the closest intersection that is outside the overstepping
        // tolerance is needed
        const auto valid_smaller = (qe.smaller() > overstep_tol);
        scalar_type t = 0.f;
        t(valid_smaller) = qe.smaller();
        t(!valid_smaller) = qe.larger();

        is = this->template build_candidate<surface_descr_t>(
            ray, mask, trf, t, mask_tolerance, mask_tol_scalor, overstep_tol);
        is.sf_desc = sf;

        return is;
    }

    /// Operator function to find intersections between a ray and a 2D cylinder
    ///
    /// @tparam mask_t is the input mask type
    ///
    /// @param ray is the input ray trajectory
    /// @param sfi the intersection to be updated
    /// @param mask is the input mask that defines the surface extent
    /// @param trf is the surface placement transform
    /// @param mask_tolerance is the tolerance for mask edges
    template <typename surface_descr_t, typename mask_t,
              typename other_algebra_t>
    DETRAY_HOST_DEVICE inline void update(
        const detail::ray<other_algebra_t> &ray,
        intersection_type<surface_descr_t> &sfi, const mask_t &mask,
        const transform3_type &trf,
        const std::array<scalar_type, 2u> &mask_tolerance = {0.f, 1.f},
        const scalar_type mask_tol_scalor = 0.f,
        const scalar_type overstep_tol = 0.f) const {
        sfi = this->operator()(ray, sfi.sf_desc, mask, trf, mask_tolerance,
                               mask_tol_scalor, overstep_tol);
    }
};

}  // namespace detray
