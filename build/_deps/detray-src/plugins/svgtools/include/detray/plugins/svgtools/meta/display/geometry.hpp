/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project inlude(s)
#include "detray/plugins/svgtools/meta/proto/eta_lines.hpp"

// Actsvg include(s)
#include "actsvg/display/geometry.hpp"

// System include(s)
#include <tuple>

namespace detray::svgtools::meta::display {

/// @brief Converts a proto eta_line to an SVG object.
inline auto eta_lines(const std::string& id,
                      const svgtools::meta::proto::eta_lines& el) {

    return actsvg::display::eta_lines(
        id, el._z, el._r,
        {std::make_tuple(el._values_main, el._stroke_main, el._show_label,
                         el._label_font),
         std::make_tuple(el._values_half, el._stroke_half, false,
                         el._label_font)});
}

}  // namespace detray::svgtools::meta::display
