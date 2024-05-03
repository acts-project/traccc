/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/cell.hpp"
#include "traccc/geometry/detector_description.hpp"

// System include(s).
#include <string_view>

namespace traccc::io::csv {

/// Read cell information from a specific CSV file
///
/// @param[out] cells       The cell collection to fill
/// @param[in]  filename    The name of the file to read
/// @param[in]  dd          The detector description to point the cells at
/// @param[in]  deduplicate Whether to deduplicate the cells
///
void read_cells(cell_collection_types::host& cells, std::string_view filename,
                const detector_description::host* dd = nullptr,
                bool deduplicate = true);

}  // namespace traccc::io::csv
