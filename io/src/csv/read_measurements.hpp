/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/alt_measurement.hpp"
#include "traccc/io/reader_edm.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <string_view>

namespace traccc::io::csv {

/// Read measurement information from a specific CSV file
///
/// @param filename The file to read the measurement data from
/// @param mr The memory resource to create the host container with
/// @return A measurement (host) container
///
measurement_reader_output read_measurements(
    std::string_view filename, vecmem::memory_resource* mr = nullptr);

}  // namespace traccc::io::csv
