/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/geometry/detector.hpp"

// System include(s).
#include <string_view>

namespace traccc::io::csv {

/// Read measurement information from a specific CSV file
///
/// @param[out] measurements The collection to fill with the measurement data
/// @param[in]  filename     The file to read the measurement data from
/// @param[in]  detector  detray detector
/// @param[in]  do_sort      Whether to sort the measurements or not
///
void read_measurements(measurement_collection_types::host& measurements,
                       std::string_view filename,
                       const traccc::default_detector::host* detector = nullptr,
                       const bool do_sort = true);

}  // namespace traccc::io::csv
