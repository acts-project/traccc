/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/io/data_format.hpp"

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/geometry/detector.hpp"

// System include(s).
#include <cstddef>
#include <string_view>

namespace traccc::io {

/// Read measurement data into memory
///
/// The file to read is selected according the naming conventions used in
/// our data.
///
/// @param[out] measurements The measurement collection to fill
/// @param[in]  event     The event ID to read in the measurements for
/// @param[in]  directory The directory holding the measurement data files
/// @param[in]  detector  detray detector
/// @param[in]  format    The format of the measurement data files (to read)
///
void read_measurements(measurement_collection_types::host& measurements,
                       std::size_t event, std::string_view directory,
                       const traccc::default_detector::host* detector = nullptr,
                       data_format format = data_format::csv);

/// Read measurement data into memory
///
/// The file name is selected explicitly by the user.
///
/// @param[out] measurements The measurement collection to fill
/// @param[in]  filename The file to read the measurement data from
/// @param[in]  detector  detray detector
/// @param[in]  format   The format of the measurement data files (to read)
///
void read_measurements(measurement_collection_types::host& measurements,
                       std::string_view filename,
                       const traccc::default_detector::host* detector = nullptr,
                       data_format format = data_format::csv);

}  // namespace traccc::io
