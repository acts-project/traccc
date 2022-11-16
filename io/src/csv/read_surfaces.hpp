/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "surface.hpp"

// Project include(s).
#include "traccc/definitions/primitives.hpp"

// System include(s).
#include <map>
#include <string_view>

namespace traccc::io::csv {

/// Read the geometry information per module and fill into a map
std::map<geometry_id, transform3> read_surfaces(std::string_view filename);

}  // namespace traccc::io::csv
