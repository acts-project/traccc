/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/track_parameters.hpp"

// System include(s).
#include <string_view>

namespace traccc::io::csv {

/// Function for track parameter writing to CSV files
///
/// @param filename The name of the file to write the data to
/// @param track_params Track parameters to write
///
void write_track_parameters(
    std::string_view filename,
    bound_track_parameters_collection_types::const_view track_params);

}  // namespace traccc::io::csv
