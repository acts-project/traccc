/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"

// System include(s).
#include <string_view>

namespace traccc::io::csv {

/// Function for cell file writing to CSV files
///
/// @param filename The name of the file to write the data to
/// @param cells    Cell collection to write
/// @param dd       Silicon detector description
/// @param use_acts_geometry_id Flag to use the ACTS geometry ID (or the Detray
///                             one)
///
void write_cells(std::string_view filename,
                 traccc::edm::silicon_cell_collection::const_view cells,
                 traccc::silicon_detector_description::const_view dd,
                 bool use_acts_geometry_id);

}  // namespace traccc::io::csv
