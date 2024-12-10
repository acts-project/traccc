/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project inlude(s)
#include "detray/plugins/svgtools/meta/proto/intersection.hpp"
#include "detray/plugins/svgtools/meta/proto/landmark.hpp"
#include "detray/plugins/svgtools/meta/proto/trajectory.hpp"

// Actsvg include(s)
#include "actsvg/core.hpp"

// System include(s)
#include <algorithm>
#include <string>
#include <vector>

namespace detray::svgtools::meta::display {

/// @brief Converts a proto landmark to a SVG object.
template <typename point3_t, typename view_t>
inline auto landmark(const std::string& id,
                     const svgtools::meta::proto::landmark<point3_t>& lm,
                     const view_t& view) {
    const auto point_view = view(std::vector{lm._position})[0];
    return actsvg::draw::marker(id, point_view, lm._marker);
}

/// @brief Converts a proto intersection to a SVG object.
template <typename point3_t, typename view_t>
inline auto intersection(
    const std::string& id,
    const svgtools::meta::proto::intersection<point3_t>& intr,
    const view_t& view) {
    actsvg::svg::object ret;
    ret._tag = "g";
    ret._id = id;
    for (size_t index = 0; index < intr._landmarks.size(); index++) {
        const auto lm = intr._landmarks[index];
        const auto svg = svgtools::meta::display::landmark(
            id + "_intersection_" + std::to_string(index), lm, view);
        ret.add_object(svg);
    }
    return ret;
}

/// @brief Converts a proto trajectory to a SVG object.
template <typename point3_t, typename view_t>
inline auto trajectory(
    const std::string& id,
    const svgtools::meta::proto::trajectory<point3_t>& p_trajectory,
    const view_t& view) {
    std::vector<actsvg::point2> points;
    auto change_view = [view](const point3_t& p) {
        return view(std::vector{p})[0];
    };
    std::ranges::transform(p_trajectory._points, std::back_inserter(points),
                           change_view);
    actsvg::svg::object ret;
    ret._tag = "g";
    ret._id = id;
    for (std::size_t i = 0; i < points.size() - 1; i++) {
        // TODO: use smooth curves instead of lines.
        const auto l = actsvg::draw::line(id, points[i], points[i + 1],
                                          p_trajectory._stroke);
        ret.add_object(l);
    }
    return ret;
}

}  // namespace detray::svgtools::meta::display
