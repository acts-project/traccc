/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/spacepoint.hpp"
#include "traccc/geometry/geometry.hpp"
#include "traccc/io/reader_edm.hpp"

// System include(s).
#include <string_view>

namespace traccc::io::csv {

/// Read spacepoint information from a specific CSV file
///
/// @param out A spacepoint & a cell_module (host) collections
/// @param filename The file to read the spacepoint data from
/// @param geom The description of the detector geometry
///
void read_spacepoints(spacepoint_reader_output& out, std::string_view filename,
                      const geometry& geom);

}  // namespace traccc::io::csv
