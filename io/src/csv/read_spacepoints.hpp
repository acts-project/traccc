/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/geometry/detector.hpp"

// System include(s).
#include <string_view>

namespace traccc::io::csv {

/// Read spacepoint information from specific CSV files
///
/// @param[out] spacepoints The spacepoint collection to fill
/// @param[out] measurements The measurement collection to fill
/// @param[in]  hit_filename  The file to read the hit/spacepoint data from
/// @param[in]  meas_filename The file to read the measurement data from
/// @param[in]  meas_hit_map_filename The file to read the mapping from
///                                   measurements to hits from
/// @param[in]  detector  detray detector
///
void read_spacepoints(edm::spacepoint_collection::host& spacepoints,
                      measurement_collection_types::host& measurements,
                      std::string_view hit_filename,
                      std::string_view meas_filename,
                      std::string_view meas_hit_map_filename,
                      const traccc::default_detector::host* detector = nullptr);

}  // namespace traccc::io::csv
