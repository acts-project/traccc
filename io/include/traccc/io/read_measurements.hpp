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
#include "traccc/geometry/detector_description.hpp"

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
/// @param[in]  event     The event ID to read in the cells for
/// @param[in]  directory The directory holding the cell data files
/// @param[in]  dd        The detector description to point the measurements at
/// @param[in]  format    The format of the measurement data files (to read)
///
void read_measurements(measurement_collection_types::host& measurements,
                       std::size_t event, std::string_view directory,
                       const detector_description::host* dd = nullptr,
                       data_format format = data_format::csv);

/// Read measurement data into memory
///
/// The file name is selected explicitly by the user.
///
/// @param[out] measurements The measurement collection to fill
/// @param[in]  filename The file to read the measurement data from
/// @param[in]  dd       The detector description to point the measurements at
/// @param[in]  format   The format of the measurement data files (to read)
///
void read_measurements(measurement_reader_output& measurements,
                       std::string_view filename,
                       const detector_description::host* dd = nullptr,
                       data_format format = data_format::csv);

}  // namespace traccc::io
