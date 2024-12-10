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
#include "detray/geometry/coordinates/line2D.hpp"
#include "detray/navigation/detail/ray.hpp"
#include "detray/navigation/intersection/intersection.hpp"

// System include(s)
#include <type_traits>

namespace detray {

template <typename frame_t, typename algebra_t, bool do_debug>
struct ray_intersector_impl;

/// A functor to find intersections between straight line and planar surface
template <concepts::soa_algebra algebra_t, bool do_debug>
struct ray_intersector_impl<line2D<algebra_t>, algebra_t, do_debug> {

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

    /// Operator function to find intersections between ray and line mask
    ///
    /// @tparam mask_t is the input mask type
    /// @tparam surface_t is the type of surface handle
    ///
    /// @param ray is the input ray trajectory
    /// @param sf the surface handle the mask is associated with
    /// @param mask is the input mask that defines the surface extent
    /// @param trf is the surface placement transform
    /// @param mask_tolerance is the tolerance for mask edges
    //
    /// @return the intersection
    template <typename surface_descr_t, typename mask_t,
              typename other_algebra_t>
    DETRAY_HOST_DEVICE inline intersection_type<surface_descr_t> operator()(
        const detail::ray<other_algebra_t> &ray, const surface_descr_t &sf,
        const mask_t &mask, const transform3_type &trf,
        const std::array<scalar_type, 2u> &mask_tolerance = {0.f, 1.f},
        const scalar_type mask_tol_scalor = 0.f,
        const scalar_type overstep_tol = 0.f) const {

        intersection_type<surface_descr_t> is;

        // line direction
        const vector3_type sz = getter::vector<3>(trf.matrix(), 0u, 2u);

        // line center
        const point3_type st = trf.translation();

        // Broadcast ray data
        const auto &pos = ray.pos();
        const auto &dir = ray.dir();
        const vector3_type ro{pos[0], pos[1], pos[2]};
        const vector3_type rd{dir[0], dir[1], dir[2]};

        // Projection of line to track direction
        const scalar_type zd = vector::dot(sz, rd);

        const scalar_type denom = 1.f - (zd * zd);

        // Case for wire is parallel to track
        if (detray::detail::all_of(denom < 1e-5f)) {
            is.status = decltype(is.status)(false);
            return is;
        }

        // vector from track position to line center
        const vector3_type t2l = st - ro;

        // t2l projection on line direction
        const scalar_type t2l_on_line = vector::dot(t2l, sz);

        // t2l projection on track direction
        const scalar_type t2l_on_track = vector::dot(t2l, rd);

        // path length to the point of closest approach on the track
        is.path = (t2l_on_track - t2l_on_line * zd) / denom;

        // point of closest approach on the track
        const point3_type m = ro + rd * is.path;
        const auto loc = mask_t::to_local_frame(trf, m, rd);
        if constexpr (intersection_type<surface_descr_t>::is_debug()) {
            is.local = loc;
        }
        is.status = mask.is_inside(
            loc, math::max(mask_tolerance[0],
                           math::min(mask_tolerance[1],
                                     mask_tol_scalor * math::fabs(is.path))));

        // Early return, in case all intersections are invalid
        if (detray::detail::none_of(is.status)) {
            return is;
        }

        is.sf_desc = sf;
        is.direction = math::signbit(is.path);
        is.volume_link = mask.volume_link();

        // Mask the values where the overstepping tolerance was not met
        is.status &= (is.path >= overstep_tol);

        return is;
    }

    /// Operator function to find intersections between a ray and a line.
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
