/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Actsvg include(s)
#include "actsvg/core.hpp"

// System include(s)
#include <string>

namespace detray::svgtools::meta::proto {

/// @brief A proto landmark class as a simple translation layer from a
/// description of a point.
template <typename point3_t>
struct landmark {
    point3_t _position{0.f, 0.f, 0.f};
    std::string _name{"unknown landmark"};
    actsvg::style::marker _marker{"x", 1.f};
};

}  // namespace detray::svgtools::meta::proto
