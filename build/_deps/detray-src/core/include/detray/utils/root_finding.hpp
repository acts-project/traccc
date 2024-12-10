/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "detray/definitions/detail/algebra.hpp"
#include "detray/definitions/detail/math.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/definitions/units.hpp"
#include "detray/utils/invalid_values.hpp"

// System include(s).
#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

namespace detray {

/// @brief Try to find a bracket around a root
///
/// @param [in] a lower initial boundary
/// @param [in] b upper initial boundary
/// @param [in] f function for which to find the root
/// @param [out] bracket bracket around the root
/// @param [in] k scale factor with which to widen the bracket at every step
///
/// @see Numerical Recepies pp. 445
///
/// @return whether a bracket was found
template <typename scalar_t, typename function_t>
DETRAY_HOST_DEVICE inline bool expand_bracket(const scalar_t a,
                                              const scalar_t b, function_t &f,
                                              std::array<scalar_t, 2> &bracket,
                                              const scalar_t k = 1.f) {

    if (a == b) {
        throw std::invalid_argument(
            "Root bracketing: Not a valid start interval [" +
            std::to_string(a) + ", " + std::to_string(b) + "]");
    }

    scalar_t lower{a > b ? b : a};
    scalar_t upper{a > b ? a : b};

    // Sample function points at interval
    scalar_t f_l{f(lower)};
    scalar_t f_u{f(upper)};
    std::size_t n_tries{0u};

    // If there is no sign change in interval, we don't know if there is a root
    while (!math::signbit(f_l * f_u)) {
        // No interval could be found to bracket the root
        // Might be correct, if there is not root
        if ((n_tries == 1000u) || !std::isfinite(f_l) || !std::isfinite(f_u)) {
#ifndef NDEBUG
            std::cout << "WARNING: Could not bracket a root" << std::endl;
#endif
            bracket = {a, b};
            return false;
        }
        scalar_t d{k * (upper - lower)};
        // Make interval larger in the direction where the function is smaller
        if (math::fabs(f_l) < math::fabs(f_u)) {
            lower -= d;
            f_l = f(lower);
        } else {
            upper += d;
            f_u = f(upper);
        }
        ++n_tries;
    }

    bracket = {lower, upper};
    return true;
}

/// @brief Find a root using the Newton-Raphson and Bisection algorithms
///
/// @param s initial guess for the root
/// @param evaluate_func evaluate the function and its derivative
/// @param max_path don't consider root if it is too far away
/// @param run_rtsafe whether to run pure Newton
///
/// @see Numerical Recepies pp. 445
///
/// @return pathlength to root and the last step size
template <typename scalar_t, typename function_t>
DETRAY_HOST_DEVICE inline std::pair<scalar_t, scalar_t> newton_raphson_safe(
    function_t &evaluate_func, scalar_t s,
    const scalar_t convergence_tolerance = 1.f * unit<scalar_t>::um,
    const std::size_t max_n_tries = 1000u,
    const scalar_t max_path = 5.f * unit<scalar_t>::m) {

    constexpr scalar_t inv{detail::invalid_value<scalar_t>()};
    constexpr scalar_t epsilon{std::numeric_limits<scalar_t>::epsilon()};

    // First, try to bracket a root
    auto f = [&evaluate_func](const scalar_t x) {
        auto [f_x, df_x] = evaluate_func(x);

        return f_x;
    };

    // Initial bracket
    scalar_t a{math::fabs(s) == 0.f ? -0.1f : 0.9f * s};
    scalar_t b{math::fabs(s) == 0.f ? 0.1f : 1.1f * s};
    std::array<scalar_t, 2> br{};
    bool is_bracketed = expand_bracket(a, b, f, br);

    // Update initial guess on the root after bracketing
    s = is_bracketed ? 0.5f * (br[1] + br[0]) : s;

    if (is_bracketed) {
        // Check bracket
        [[maybe_unused]] auto [f_a, df_a] = evaluate_func(br[0]);
        [[maybe_unused]] auto [f_b, df_b] = evaluate_func(br[1]);

        assert(math::signbit(f_a * f_b) && "Incorrect bracket around root");

        // Root is not within the maximal pathlength
        bool bracket_outside_tol{s > max_path &&
                                 ((br[0] < -max_path && br[1] < -max_path) ||
                                  (br[0] > max_path && br[1] > max_path))};
        if (bracket_outside_tol) {
#ifndef NDEBUG
            std::cout << "INFO: Root outside maximum search area - skipping"
                      << std::endl;
#endif
            return std::make_pair(inv, inv);
        }

        // Root already found?
        if (math::fabs(f_a) < convergence_tolerance) {
            return std::make_pair(a, epsilon);
        }
        if (math::fabs(f_b) < convergence_tolerance) {
            return std::make_pair(b, epsilon);
        }

        // Make 'a' the boundary for the negative function value -> easier to
        // update
        bool is_lower_a{math::signbit(f_a)};
        a = br[is_lower_a ? 0u : 1u];
        b = br[is_lower_a ? 1u : 0u];
    }

    // Run the iteration on s
    scalar_t s_prev{0.f};
    std::size_t n_tries{0u};
    auto [f_s, df_s] = evaluate_func(s);
    if (math::fabs(f_s) < convergence_tolerance) {
        return std::make_pair(s, epsilon);
    }
    if (math::signbit(f_s)) {
        a = s;
    } else {
        b = s;
    }

    while (math::fabs(s - s_prev) > convergence_tolerance) {

        // Does Newton step escape bracket?
        bool bracket_escape{true};
        scalar_t s_newton{0.f};
        if (math::fabs(df_s) != 0.f) {
            s_newton = s - f_s / df_s;
            bracket_escape = math::signbit((s_newton - a) * (b - s_newton));
        }

        // This criterion from Numerical Recipes seems to work, but why?
        /*const bool slow_convergence{math::fabs(2.f * f_s) >
                                    math::fabs((s_prev - s) * df_s)};*/

        // Take a bisection step if it converges faster than Newton
        // |f(next_newton_s)| > |f(next_bisection_s)|
        bool slow_convergence{true};
        // The criterion is only well defined if the step lengths are small
        if (const scalar_t ds_bisection{0.5f * (a + b) - s};
            is_bracketed &&
            (math::fabs(ds_bisection) < 10.f * unit<scalar_t>::mm)) {
            slow_convergence =
                (2.f * math::fabs(f_s) > math::fabs(df_s * ds_bisection + f_s));
        }

        s_prev = s;

        // Run bisection if Newton-Raphson would be poor
        if (is_bracketed &&
            (bracket_escape || slow_convergence || math::fabs(df_s) == 0.f)) {
            // Test the function sign in the middle of the interval
            s = 0.5f * (a + b);
        } else {
            // No intersection can be found if dividing by zero
            if (!is_bracketed && math::fabs(df_s) == 0.f) {
                std::cout << "WARNING: Encountered invalid derivative "
                          << std::endl;

                return std::make_pair(inv, inv);
            }

            s = s_newton;
        }

        // Update function and bracket
        std::tie(f_s, df_s) = evaluate_func(s);
        if (is_bracketed && math::signbit(f_s)) {
            a = s;
        } else {
            b = s;
        }

        // Converges to a point outside the search space
        if (math::fabs(s) > max_path && math::fabs(s_prev) > max_path &&
            ((a < -max_path && b < -max_path) ||
             (a > max_path && b > max_path))) {
#ifndef NDEBUG
            std::cout << "WARNING: Root finding left the search space"
                      << std::endl;
#endif
            return std::make_pair(inv, inv);
        }

        ++n_tries;
        // No intersection found within max number of trials
        if (n_tries >= max_n_tries) {

            // Should have found the root
            if (is_bracketed) {
                throw std::runtime_error(
                    "ERROR: Helix intersector did not "
                    "find root for s=" +
                    std::to_string(s) + " in [" + std::to_string(a) + ", " +
                    std::to_string(b) + "]");
            } else {
#ifndef NDEBUG
                std::cout << "WARNING: Helix intersector did not "
                             "converge after "
                          << n_tries << " steps unbracketed search"
                          << std::endl;
#endif
            }
            return std::make_pair(inv, inv);
        }
    }
    //  Final pathlengt to root and latest step size
    return std::make_pair(s, math::fabs(s - s_prev));
}

/// @brief Fill an intersection with the result of the root finding
///
/// @param [in] traj the test trajectory that intersects the surface
/// @param [out] sfi the surface intersection
/// @param [in] s path length to the root
/// @param [in] ds approximation error for the root
/// @param [in] mask the mask of the surface
/// @param [in] trf the transform of the surface
/// @param [in] mask_tolerance minimal and maximal mask tolerance
template <typename scalar_t, typename intersection_t, typename surface_descr_t,
          typename mask_t, typename trajectory_t, typename transform3_t>
DETRAY_HOST_DEVICE inline void build_intersection(
    const trajectory_t &traj, intersection_t &sfi, const scalar_t s,
    const scalar_t ds, const surface_descr_t sf_desc, const mask_t &mask,
    const transform3_t &trf, const std::array<scalar_t, 2> &mask_tolerance) {

    // Build intersection struct from test trajectory, if the distance is valid
    if (!detail::is_invalid_value(s)) {
        sfi.path = s;
        sfi.local = mask_t::to_local_frame(trf, traj.pos(s), traj.dir(s));
        const scalar_t cos_incidence_angle = vector::dot(
            mask_t::get_local_frame().normal(trf, sfi.local), traj.dir(s));

        scalar_t tol{mask_tolerance[1]};
        if (detail::is_invalid_value(tol)) {
            // Due to floating point errors this can be negative if cos ~ 1
            const scalar_t sin_inc2{
                math::fabs(1.f - cos_incidence_angle * cos_incidence_angle)};

            tol = math::fabs(ds * math::sqrt(sin_inc2));
        }
        sfi.status = mask.is_inside(sfi.local, tol);
        sfi.sf_desc = sf_desc;
        sfi.direction = !math::signbit(s);
        sfi.volume_link = mask.volume_link();
    } else {
        // Not a valid intersection
        sfi.status = false;
    }
}

}  // namespace detray
