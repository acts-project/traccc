/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/navigation/intersection/helix_intersector.hpp"
#include "detray/navigation/intersection/ray_intersector.hpp"

namespace detray {

/// @brief Intersection interface for detector surfaces.
///
/// Composes the different intersector options into a unifyed interface
template <typename shape_t, typename algebra_t, bool do_debug = false>
struct intersector {

    using algebra_type = algebra_t;
    using scalar_type = dscalar<algebra_t>;
    using transform3_type = dtransform3D<algebra_t>;

    /// How to intersect surfaces with rays
    using ray_intersector_type = ray_intersector<shape_t, algebra_t, do_debug>;

    /// How to intersect surfaces with helices
    using helix_intersector_type =
        helix_intersector<shape_t, algebra_t, do_debug>;

    // Test with int as dummy surface descriptor type
    static_assert(
        std::same_as<
            typename ray_intersector_type::template intersection_type<int>,
            typename helix_intersector_type::template intersection_type<int>>);

    /// @returns the intersection(s) between a surface and the ray @param ray
    template <typename surface_descr_t, typename mask_t>
    DETRAY_HOST_DEVICE inline decltype(auto) operator()(
        const detail::ray<algebra_t> &ray, const surface_descr_t &sf,
        const mask_t &mask, const transform3_type &trf,
        const std::array<scalar_type, 2u> mask_tolerance =
            {0.f, 1.f * unit<scalar_type>::mm},
        const scalar_type mask_tol_scalor = 0.f,
        const scalar_type overstep_tol = 0.f) const {

        return ray_intersector_type{}(ray, sf, mask, trf, mask_tolerance,
                                      mask_tol_scalor, overstep_tol);
    }

    /// @returns the intersection(s) between a surface and the helix @param h
    template <typename surface_descr_t, typename mask_t>
    DETRAY_HOST_DEVICE inline decltype(auto) operator()(
        const detail::helix<algebra_t> &h, const surface_descr_t &sf,
        const mask_t &mask, const transform3_type &trf,
        const std::array<scalar_type, 2u> mask_tolerance =
            {0.f, 1.f * unit<scalar_type>::mm},
        const scalar_type = 0.f, const scalar_type = 0.f) const {

        return helix_intersector_type{}(h, sf, mask, trf, mask_tolerance);
    }
};

}  // namespace detray
