/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/math.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/geometry/coordinates/line2D.hpp"
#include "detray/navigation/detail/helix.hpp"
#include "detray/navigation/intersection/intersection.hpp"

// System include(s)
#include <iostream>
#include <type_traits>

namespace detray {

template <typename frame_t, typename algebra_t>
struct helix_intersector_impl;

/// @brief Intersection implementation for helical trajectories with line
/// surfaces.
///
/// The algorithm uses the Newton-Raphson method to find an intersection on
/// the unbounded surface and then applies the mask.
template <concepts::aos_algebra algebra_t>
struct helix_intersector_impl<line2D<algebra_t>, algebra_t> {

    using scalar_type = dscalar<algebra_t>;
    using point3_type = dpoint3D<algebra_t>;
    using vector3_type = dvector3D<algebra_t>;
    using transform3_type = dtransform3D<algebra_t>;

    template <typename surface_descr_t>
    using intersection_type = intersection2D<surface_descr_t, algebra_t, true>;
    using helix_type = detail::helix<algebra_t>;

    /// Operator function to find intersections between helix and line mask
    ///
    /// @tparam mask_t is the input mask type
    /// @tparam surface_desc_t is the input surface descriptor type
    ///
    /// @param h is the input helix trajectory
    /// @param sf_desc is the surface descriptor
    /// @param mask is the input mask
    /// @param trf is the transform
    /// @param mask_tolerance is the tolerance for mask edges
    ///
    /// @return the intersection
    template <typename surface_descr_t, typename mask_t>
    DETRAY_HOST_DEVICE inline intersection_type<surface_descr_t> operator()(
        const helix_type &h, const surface_descr_t &sf_desc, const mask_t &mask,
        const transform3_type &trf,
        const std::array<scalar_type, 2u> mask_tolerance =
            {detail::invalid_value<scalar_type>(),
             detail::invalid_value<scalar_type>()},
        const scalar_type = 0.f, const scalar_type = 0.f) const {

        assert((mask_tolerance[0] == mask_tolerance[1]) &&
               "Helix intersectors use only one mask tolerance value");

        intersection_type<surface_descr_t> sfi{};

        if (!run_rtsafe) {
            // line axis direction
            const vector3_type l = getter::vector<3>(trf.matrix(), 0u, 2u);

            // line center
            const point3_type c = trf.translation();

            // initial track direction
            const vector3_type t0 = h.dir(0.f);

            // initial track position
            const point3_type r0 = h.pos(0.f);

            // Projection of line to track direction
            const scalar_type lt0{vector::dot(l, t0)};

            const scalar_type denom{1.f - (lt0 * lt0)};

            // Case for wire is parallel to track
            // @NOTE We might not have to call this which is meant to be for ray
            // intersection...
            if (denom < 1e-5f) {
                sfi.status = false;
                return sfi;
            }

            // vector from track position to line center
            const vector3_type D = c - r0;

            // D projection on line direction
            const scalar_type P{vector::dot(D, l)};

            // D projection on track direction
            const scalar_type Q{vector::dot(D, t0)};

            // Path length to the point of closest approach on the track
            // @NOTE Ray intersection algorithm is used for the initial guess on
            // the path length
            scalar_type s{1.f / denom * (Q - P * lt0)};
            scalar_type s_prev{0.f};

            // Run the iteration on s
            std::size_t n_tries{0u};
            while (math::fabs(s - s_prev) > convergence_tolerance &&
                   n_tries < max_n_tries) {

                // track direction
                const vector3_type t = h.dir(s);

                // track position
                const point3_type r = h.pos(s);

                // Projection of (track position - center) to the line
                const scalar_type A = vector::dot(r - c, l);

                // Vector orthogonal to the line and passing the track position
                // w = r - (c + ((r - c) * l)l)
                const vector3_type w = r - (c + A * l);

                // f(s) = t * w = 0
                const scalar_type f = vector::dot(t, w);

                // dtds = d^2r/ds^2 = qop * (t X b_field)
                const vector3_type dtds =
                    h.qop() * vector::cross(t, *h._mag_field);
                // dwds = t - (t * l)l
                const vector3_type dwds = t - vector::dot(t, l) * l;

                // f'(s) = dtds * w + t * dwds
                const scalar_type dfds =
                    vector::dot(dtds, w) + vector::dot(t, dwds);

                // x_n+1 = x_n - f(s) / f'(s)
                s_prev = s;
                s -= f / dfds;

                ++n_tries;
            }

            // No intersection found within max number of trials
            if (n_tries == max_n_tries) {
                return sfi;
            }

            // Build intersection struct from helix parameters
            sfi.path = s;
            sfi.local = mask_t::to_local_frame(trf, h.pos(s), h.dir(s));
            const scalar_type cos_incidence_angle = vector::dot(
                mask_t::get_local_frame().normal(trf, sfi.local), h.dir(s));
            scalar_type tol{mask_tolerance[1]};
            if (detail::is_invalid_value(tol)) {
                // Due to floating point errors this can be negative if cos ~ 1
                const scalar_type sin_inc2{math::fabs(
                    1.f - cos_incidence_angle * cos_incidence_angle)};

                tol = math::fabs((s - s_prev) * math::sqrt(sin_inc2));
            }
            sfi.status = mask.is_inside(sfi.local, tol);
            sfi.sf_desc = sf_desc;
            sfi.direction = !math::signbit(s);
            sfi.volume_link = mask.volume_link();

            return sfi;
        } else {
            // line axis direction
            const vector3_type l = getter::vector<3>(trf.matrix(), 0u, 2u);

            // line center
            const point3_type c = trf.translation();

            // initial track direction
            const vector3_type t0 = h.dir(0.f);

            // initial track position
            const point3_type r0 = h.pos(0.f);

            // Projection of line to track direction
            const scalar_type lt0{vector::dot(l, t0)};

            const scalar_type denom{1.f - (lt0 * lt0)};

            // Case for wire is parallel to track
            // @NOTE We might not have to call this which is meant to be for ray
            // intersection...
            if (denom < 1e-5f) {
#ifndef NDEBUG
                std::cout << "ERROR: Helix line intersector encountered "
                             "invalid value!"
                          << std::endl;
#endif
                sfi.status = false;
                return sfi;
            }

            // vector from track position to line center
            const vector3_type D = c - r0;

            // D projection on line direction
            const scalar_type P{vector::dot(D, l)};

            // D projection on track direction
            const scalar_type Q{vector::dot(D, t0)};

            // Path length to the point of closest approach on the track
            // @NOTE Ray intersection algorithm is used for the initial guess on
            // the path length
            scalar_type s_ini{1.f / denom * (Q - P * lt0)};

            /// Evaluate the function and its derivative at the point @param x
            auto line_inters_func = [&h, &c, &l](const scalar_type x) {
                // track direction
                const vector3_type t = h.dir(x);

                // track position
                const point3_type r = h.pos(x);

                // Projection of (track position - center) to the line
                const scalar_type A = vector::dot(r - c, l);

                // Vector orthogonal to the line and passing the track position
                // w = r - (c + ((r - c) * l)l)
                const vector3_type w = r - (c + A * l);

                // f(s) = t * w = 0
                const scalar_type f = vector::dot(t, w);

                // dtds = d^2r/ds^2 = qop * (t X b_field)
                const vector3_type dtds =
                    h.qop() * vector::cross(t, *h._mag_field);
                // dwds = t - (t * l)l
                const vector3_type dwds = t - vector::dot(t, l) * l;

                // f'(s) = dtds * w + t * dwds
                const scalar_type dfds =
                    vector::dot(dtds, w) + vector::dot(t, dwds);

                return std::make_tuple(f, dfds);
            };

            // Run the root finding algorithm
            const auto [s, ds] = newton_raphson_safe(line_inters_func, s_ini,
                                                     convergence_tolerance,
                                                     max_n_tries, max_path);

            // Build intersection struct from the root
            build_intersection(h, sfi, s, ds, sf_desc, mask, trf,
                               mask_tolerance);

            return sfi;
        }
    }

    /// Interface to use fixed mask tolerance
    template <typename surface_descr_t, typename mask_t>
    DETRAY_HOST_DEVICE inline intersection_type<surface_descr_t> operator()(
        const helix_type &h, const surface_descr_t &sf_desc, const mask_t &mask,
        const transform3_type &trf, const scalar_type mask_tolerance,
        const scalar_type = 0.f, const scalar_type = 0.f) const {
        return this->operator()(h, sf_desc, mask, trf,
                                {mask_tolerance, mask_tolerance}, 0.f);
    }

    /// Tolerance for convergence
    scalar_type convergence_tolerance{1.f * unit<scalar_type>::um};
    // Guard against inifinite loops
    std::size_t max_n_tries{1000u};
    // Early exit, if the intersection is too far away
    scalar_type max_path{5.f * unit<scalar_type>::m};
    // Complement the Newton algorithm with Bisection steps
    bool run_rtsafe{true};
};

}  // namespace detray
