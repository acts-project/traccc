/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/navigation/detail/ray.hpp"
#include "detray/plugins/svgtools/meta/proto/trajectory.hpp"
#include "detray/plugins/svgtools/styling/styling.hpp"

// System include(s)
#include <vector>

namespace detray::svgtools::conversion {

/// @returns The proto trajectory of a vector of points.
template <typename point3_t>
inline auto trajectory(const std::vector<point3_t>& points,
                       const styling::trajectory_style& style =
                           styling::svg_default::trajectory_style) {

    using p_trajectory_t = svgtools::meta::proto::trajectory<point3_t>;
    p_trajectory_t p_trajectory;

    p_trajectory._points = points;

    svgtools::styling::apply_style(p_trajectory, style);

    return p_trajectory;
}

/// @param traj the trajectory
/// @param path_legth the length of the path
/// @param step_size the step size, ie., ds
/// @param style the style settings
///
/// @returns The proto trajectory of a parametrized trajectory
template <template <typename> class trajectory_t, typename algebra_t>
inline auto trajectory(const trajectory_t<algebra_t>& traj,
                       const styling::trajectory_style& style =
                           styling::svg_default::trajectory_style,
                       const dscalar<algebra_t> path_length = 500.f,
                       const dscalar<algebra_t> step_size = 1.f) {

    const dscalar<algebra_t> S0 = 0.f;
    std::vector<dpoint3D<algebra_t>> points;
    for (auto s = S0; s < path_length; s += step_size) {
        points.push_back(traj.pos(s));
    }

    return trajectory(points, style);
}

/// @param traj the trajectory
/// @param path_legth the length of the path
/// @param style the style settings
///
/// @returns The proto trajectory of a ray.
template <typename algebra_t>
inline auto trajectory(const detray::detail::ray<algebra_t>& traj,
                       const styling::trajectory_style& style =
                           styling::svg_default::trajectory_style,
                       const dscalar<algebra_t> path_length = 500.f) {

    const dscalar<algebra_t> S0 = 0.f;
    std::vector<dpoint3D<algebra_t>> points = {traj.pos(S0),
                                               traj.pos(path_length)};

    return trajectory(points, style);
}

}  // namespace detray::svgtools::conversion
