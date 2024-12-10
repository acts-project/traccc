/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/math.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/definitions/units.hpp"
#include "detray/geometry/coordinates/cylindrical2D.hpp"
#include "detray/navigation/detail/ray.hpp"
#include "detray/navigation/intersection/intersection.hpp"
#include "detray/utils/invalid_values.hpp"
#include "detray/utils/quadratic_equation.hpp"

// System include(s)
#include <type_traits>

namespace detray {

/// A functor to find intersections between trajectory and concentric cylinder
/// mask
template <concepts::aos_algebra algebra_t, bool do_debug = false>
struct ray_concentric_cylinder_intersector {

    /// linear algebra types
    /// @{
    using scalar_type = dscalar<algebra_t>;
    using point3_type = dpoint3D<algebra_t>;
    using vector3_type = dvector3D<algebra_t>;
    using transform3_type = dtransform3D<algebra_t>;
    /// @}

    template <typename surface_descr_t>
    using intersection_type =
        intersection2D<surface_descr_t, algebra_t, do_debug>;
    using ray_type = detail::ray<algebra_t>;

    /// Operator function to find intersections between ray and concentric
    /// cylinder mask
    ///
    /// @tparam mask_t is the input mask type
    /// @tparam surface_descr_t is the type of surface handle
    ///
    /// @param ray is the input ray trajectory
    /// @param sf the surface handle the mask is associated with
    /// @param mask is the input mask that defines the surface extent
    /// @param trf is the surface placement transform
    /// @param mask_tolerance is the tolerance for mask edges
    /// @param overstep_tol negative cutoff for the path
    ///
    /// @return the intersection
    template <typename surface_descr_t, typename mask_t>
    DETRAY_HOST_DEVICE inline intersection_type<surface_descr_t> operator()(
        const ray_type &ray, const surface_descr_t &sf, const mask_t &mask,
        const transform3_type & /*trf*/,
        const std::array<scalar_type, 2u> mask_tolerance =
            {0.f, 1.f * unit<scalar_type>::mm},
        const scalar_type mask_tol_scalor = 0.f,
        const scalar_type overstep_tol = 0.f) const {

        intersection_type<surface_descr_t> is;

        const scalar_type r{mask[mask_t::shape::e_r]};
        // Two points on the line, these are in the cylinder frame
        const point3_type &ro = ray.pos();
        const vector3_type &rd = ray.dir();
        const point3_type &l0 = ro;
        const point3_type l1 = ro + rd;

        // swap coorinates x/y for numerical stability
        const bool swap_x_y = math::fabs(rd[0]) < 1e-3f;

        unsigned int _x = swap_x_y ? 1u : 0u;
        unsigned int _y = swap_x_y ? 0u : 1u;
        const scalar_type k{(l0[_y] - l1[_y]) / (l0[_x] - l1[_x])};
        const scalar_type d{l1[_y] - k * l1[_x]};

        detail::quadratic_equation<scalar_type> qe{(1.f + k * k), 2.f * k * d,
                                                   d * d - r * r};

        if (qe.solutions() > 0) {
            const scalar_type overstep_tolerance{overstep_tol};
            std::array<point3_type, 2> candidates;
            std::array<scalar_type, 2> t01 = {0.f, 0.f};

            candidates[0][_x] = qe.smaller();
            candidates[0][_y] = k * qe.smaller() + d;
            t01[0] = (candidates[0][_x] - ro[_x]) / rd[_x];
            candidates[0][2] = ro[2] + t01[0] * rd[2];

            candidates[1][_x] = qe.larger();
            candidates[1][_y] = k * qe.larger() + d;
            t01[1] = (candidates[1][_x] - ro[_x]) / rd[_x];
            candidates[1][2] = ro[2] + t01[1] * rd[2];

            // Chose the index, take the smaller positive one
            const unsigned int cindex =
                (t01[0] < t01[1] && t01[0] > overstep_tolerance)
                    ? 0u
                    : (t01[0] < overstep_tolerance &&
                               t01[1] > overstep_tolerance
                           ? 1u
                           : 0u);
            if (t01[0] > overstep_tolerance || t01[1] > overstep_tolerance) {

                const point3_type p3 = candidates[cindex];
                const scalar_type phi{getter::phi(p3)};
                const point3_type loc{r * phi, p3[2], r};
                if constexpr (intersection_type<surface_descr_t>::is_debug()) {
                    is.local = loc;
                }

                is.path = t01[cindex];
                // In this case, the point has to be in cylinder3 coordinates
                // for the r-check
                // Tolerance: per mille of the distance
                is.status = mask.is_inside(
                    loc, math::max(
                             mask_tolerance[0],
                             math::min(mask_tolerance[1],
                                       mask_tol_scalor * math::fabs(is.path))));
                is.sf_desc = sf;
                is.direction = !detail::signbit(is.path);
                is.volume_link = mask.volume_link();
            }
        }
        return is;
    }

    /// Interface to use fixed mask tolerance
    template <typename surface_descr_t, typename mask_t>
    DETRAY_HOST_DEVICE inline intersection_type<surface_descr_t> operator()(
        const ray_type &ray, const surface_descr_t &sf, const mask_t &mask,
        const transform3_type &trf, const scalar_type mask_tolerance,
        const scalar_type overstep_tol = 0.f) const {
        return this->operator()(ray, sf, mask, trf, {mask_tolerance, 0.f}, 0.f,
                                overstep_tol);
    }

    /// Operator function to find intersections between a ray and a 2D cylinder
    ///
    /// @tparam mask_t is the input mask type
    /// @tparam surface_descr_t is the type of surface handle
    ///
    /// @param ray is the input ray trajectory
    /// @param sfi the intersection to be updated
    /// @param mask is the input mask that defines the surface extent
    /// @param trf is the surface placement transform
    /// @param mask_tolerance is the tolerance for mask edges
    /// @param overstep_tol negative cutoff for the path
    template <typename surface_descr_t, typename mask_t>
    DETRAY_HOST_DEVICE inline void update(
        const ray_type &ray, intersection_type<surface_descr_t> &sfi,
        const mask_t &mask, const transform3_type &trf,
        const std::array<scalar_type, 2u> &mask_tolerance =
            {0.f, 1.f * unit<scalar_type>::mm},
        const scalar_type mask_tol_scalor = 0.f,
        const scalar_type overstep_tol = 0.f) const {
        sfi = this->operator()(ray, sfi.sf_desc, mask, trf, mask_tolerance,
                               mask_tol_scalor, overstep_tol)[0];
    }
};

}  // namespace detray
