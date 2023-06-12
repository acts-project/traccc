/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/alt_measurement.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/io/reader_edm.hpp"

// System include(s).
#include <string_view>

namespace traccc::io::csv {

/// Read measurement information from a specific CSV file
///
/// @param out A measurement & a cell_module (host) collections
/// @param filename The file to read the measurement data from
///
void read_measurements(measurement_reader_output& out,
                       std::string_view filename);

measurement_container_types::host read_measurements_container(
    std::string_view filename, vecmem::memory_resource* mr = nullptr);

}  // namespace traccc::io::csv
