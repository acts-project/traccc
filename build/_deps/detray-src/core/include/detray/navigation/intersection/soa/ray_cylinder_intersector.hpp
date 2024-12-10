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
#include "detray/geometry/coordinates/cylindrical2D.hpp"
#include "detray/navigation/detail/ray.hpp"
#include "detray/navigation/intersection/intersection.hpp"
#include "detray/utils/quadratic_equation.hpp"

// System include(s)
#include <type_traits>

namespace detray {

template <typename frame_t, typename algebra_t, bool do_debug>
struct ray_intersector_impl;

/// A functor to find intersections between straight line and planar surface
template <concepts::soa_algebra algebra_t, bool do_debug>
struct ray_intersector_impl<cylindrical2D<algebra_t>, algebra_t, do_debug> {

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

    /// Operator function to find intersections between a ray and a 2D cylinder
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
    /// @return the intersections.
    template <typename surface_descr_t, typename mask_t,
              typename other_algebra_t>
    DETRAY_HOST_DEVICE inline std::array<intersection_type<surface_descr_t>, 2>
    operator()(const detail::ray<other_algebra_t> &ray,
               const surface_descr_t &sf, const mask_t &mask,
               const transform3_type &trf,
               const std::array<scalar_type, 2u> &mask_tolerance = {0.f, 1.f},
               const scalar_type mask_tol_scalor = 0.f,
               const scalar_type overstep_tol = 0.f) const {

        // One or both of these solutions might be invalid
        const auto qe = solve_intersection(ray, mask, trf);

        std::array<intersection_type<surface_descr_t>, 2> ret;
        ret[1] = build_candidate<surface_descr_t>(
            ray, mask, trf, qe.larger(), mask_tolerance, mask_tol_scalor,
            overstep_tol);
        ret[1].sf_desc = sf;

        ret[0] = build_candidate<surface_descr_t>(
            ray, mask, trf, qe.smaller(), mask_tolerance, mask_tol_scalor,
            overstep_tol);
        ret[0].sf_desc = sf;

        // Even if there are two geometrically valid solutions, the smaller one
        // might not be passed on if it is below the overstepping tolerance:
        // see 'build_candidate'
        return ret;
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
    template <typename mask_t, typename surface_descr_t,
              typename other_algebra_t>
    DETRAY_HOST_DEVICE inline void update(
        const detail::ray<other_algebra_t> &ray,
        intersection_type<surface_descr_t> &sfi, const mask_t &mask,
        const transform3_type &trf,
        const std::array<scalar_type, 2u> &mask_tolerance = {0.f, 1.f},
        const scalar_type mask_tol_scalor = 0.f,
        const scalar_type overstep_tol = 0.f) const {

        // One or both of these solutions might be invalid
        const auto qe = solve_intersection(ray, mask, trf);

        // Construct the candidate only when needed
        sfi.status = (qe.solutions() > 0.f);

        if (detray::detail::none_of(sfi.status)) {
            return;
        }

        sfi = build_candidate<surface_descr_t>(ray, mask, trf, qe.smaller(),
                                               mask_tolerance, mask_tol_scalor,
                                               overstep_tol);
    }

    protected:
    /// Calculates the distance to the (two) intersection points on the
    /// cylinder in global coordinates.
    ///
    /// @returns a quadratic equation object that contains the solution(s).
    template <typename mask_t, typename other_algebra_t>
    DETRAY_HOST_DEVICE inline auto solve_intersection(
        const detail::ray<other_algebra_t> &ray, const mask_t &mask,
        const transform3_type &trf) const {
        const auto &m = trf.matrix();
        const vector3_type sz = getter::vector<3>(m, 0u, 2u);
        const vector3_type sc = getter::vector<3>(m, 0u, 3u);

        const scalar_type r = mask[mask_t::shape::e_r];

        const auto &pos = ray.pos();
        const auto &dir = ray.dir();
        const point3_type ro{pos[0], pos[1], pos[2]};
        const vector3_type rd{dir[0], dir[1], dir[2]};

        const vector3_type tmp = ro - sc;
        const auto pc_cross_sz = vector::cross(tmp, sz);
        const auto rd_cross_sz = vector::cross(rd, sz);
        const scalar_type a = vector::dot(rd_cross_sz, rd_cross_sz);
        const scalar_type b = 2.f * vector::dot(rd_cross_sz, pc_cross_sz);
        const scalar_type c = vector::dot(pc_cross_sz, pc_cross_sz) - (r * r);

        return detail::quadratic_equation<scalar_type>{a, b, c};
    }

    /// From the intersection path, construct an intersection candidate and
    /// check it against the surface boundaries (mask).
    ///
    /// @returns the intersection candidate. Might be (partially) uninitialized
    /// if the overstepping tolerance is not met or the intersection lies
    /// outside of the mask.
    template <typename surface_descr_t, typename mask_t,
              typename other_algebra_t>
    DETRAY_HOST_DEVICE inline intersection_type<surface_descr_t>
    build_candidate(const detail::ray<other_algebra_t> &ray, const mask_t &mask,
                    const transform3_type &trf, const scalar_type path,
                    const std::array<scalar_type, 2u> &mask_tolerance =
                        {0.f, 1.f * unit<scalar_type>::mm},
                    const scalar_type mask_tol_scalor = 0.f,
                    const scalar_type overstep_tol = 0.f) const {

        intersection_type<surface_descr_t> is;

        const auto &pos = ray.pos();
        const auto &dir = ray.dir();
        const point3_type ro{pos[0], pos[1], pos[2]};
        const vector3_type rd{dir[0], dir[1], dir[2]};

        is.path = path;
        const point3_type p3 = ro + is.path * rd;

        const auto loc = mask_t::to_local_frame(trf, p3);
        if constexpr (intersection_type<surface_descr_t>::is_debug()) {
            is.local = loc;
        }
        is.status = mask.is_inside(
            loc, math::max(mask_tolerance[0],
                           math::min(mask_tolerance[1],
                                     mask_tol_scalor * math::fabs(is.path))));

        is.direction = !math::signbit(is.path);
        is.volume_link = mask.volume_link();

        // Mask the values where the overstepping tolerance was not met
        is.status &= (is.path >= overstep_tol);

        return is;
    }
};

}  // namespace detray
