/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Actsvg include(s)
#include "actsvg/core.hpp"

// System include(s)
#include <string>
#include <vector>

namespace detray::svgtools::utils {

template <typename iterator_t>
inline auto group(const std::string& identification,
                  const iterator_t& iterator) {
    actsvg::svg::object ret;
    ret._tag = "g";
    ret._id = identification;
    for (const auto& item : iterator) {
        ret.add_object(item);
    }
    return ret;
}

inline auto group(const std::string& identification) {
    return group(identification, std::vector<actsvg::svg::object>{});
}
}  // namespace detray::svgtools::utils
