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
#include "traccc/edm/spacepoint.hpp"
#include "traccc/geometry/detector.hpp"

// System include(s).
#include <cstddef>
#include <string_view>

namespace traccc::io {

/// Read spacepoint data into memory
///
/// The file to read is selected according the naming conventions used in
/// our data.
///
/// @param[out] spacepoints The spacepoint collection to fill
/// @param[in]  event     The event ID to read in the spacepoints for
/// @param[in]  directory The directory holding the spacepoint data files
/// @param[in]  detector  detray detector
/// @param[in]  format    The format of the data files (to read)
///
void read_spacepoints(spacepoint_collection_types::host& spacepoints,
                      std::size_t event, std::string_view directory,
                      const traccc::default_detector::host* detector = nullptr,
                      data_format format = data_format::csv);

/// Read spacepoint data into memory
///
/// The file name is selected explicitly by the user.
///
/// @param[out] spacepoints The spacepoint collection to fill
/// @param[in]  hit_filename  The file to read the hit/spacepoint data from
/// @param[in]  meas_filename The file to read the measurement data from
/// @param[in]  meas_hit_map_filename The file to read the mapping from
///                                   measurements to hits from
/// @param[in]  detector  detray detector
/// @param[in]  format The format of the data files (to read)
///
void read_spacepoints(spacepoint_collection_types::host& spacepoints,
                      std::string_view hit_filename,
                      std::string_view meas_filename,
                      std::string_view meas_hit_map_filename,
                      const traccc::default_detector::host* detector = nullptr,
                      data_format format = data_format::csv);

}  // namespace traccc::io
