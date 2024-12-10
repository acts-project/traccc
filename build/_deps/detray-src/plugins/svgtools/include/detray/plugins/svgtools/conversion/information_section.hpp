/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/geometry/tracking_surface.hpp"
#include "detray/plugins/svgtools/meta/proto/information_section.hpp"

// Actsvg include(s)
#include "actsvg/core.hpp"
#include "actsvg/meta.hpp"

// System include(s)
#include <array>
#include <iomanip>
#include <sstream>
#include <string>

namespace detray::svgtools::conversion {

/// @returns a point as a string.
template <typename point3_t>
inline std::string point_to_string(point3_t point) {

    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << "(" << point[0] << ", "
           << point[1] << ", " << point[2] << ")";
    return stream.str();
}

/// @returns the information section for a detray surface.
template <typename detector_t>
inline auto information_section(
    const typename detector_t::geometry_context& context,
    const detray::tracking_surface<detector_t>& d_surface) {

    using point3_t = typename detector_t::point3_type;

    svgtools::meta::proto::information_section<point3_t> is;
    is._title = d_surface.is_portal() ? "Portal" : "Surface";
    const auto position =
        d_surface.transform(context).point_to_global(d_surface.centroid());

    is._info = {"Idx: " + std::to_string(d_surface.index()),
                point_to_string(position)};
    is._position = position;

    return is;
}

}  // namespace detray::svgtools::conversion
