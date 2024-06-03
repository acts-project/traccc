/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/io/reader_edm.hpp"

// System include(s).
#include <cstdint>
#include <map>
#include <string_view>

namespace traccc::io::csv {

/// Read measurement information from a specific CSV file
///
/// @param out A measurement & a cell_module (host) collections
/// @param filename The file to read the measurement data from
/// @param do_sort Whether to sort the measurements or not
/// @param bardoce_map An object to perform barcode re-mapping with
///                    (For Acts->Detray identifier re-mapping, if necessary)
///
void read_measurements(measurement_reader_output& out,
                       std::string_view filename, const bool do_sort = true,
                       const std::map<std::uint64_t, detray::geometry::barcode>*
                           barcode_map = nullptr);

}  // namespace traccc::io::csv
