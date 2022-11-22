/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/io/data_format.hpp"

// Project include(s).
#include "traccc/definitions/primitives.hpp"

// System include(s).
#include <map>
#include <string_view>

namespace traccc::io::details {

/// Read the geometry information for the modules into a map
///
/// @param filename The file to read the information from
/// @param format The format of the input file
/// @return A map of transformations for each geometry ID
///
std::map<geometry_id, transform3> read_surfaces(
    std::string_view filename, data_format format = data_format::csv);

}  // namespace traccc::io::details
