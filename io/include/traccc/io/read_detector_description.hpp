/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/io/data_format.hpp"

// Project include(s).
#include "traccc/geometry/silicon_detector_description.hpp"

// Detray include(s).
#include <detray/geometry/barcode.hpp>

// System include(s).
#include <cstdint>
#include <map>
#include <string_view>

namespace traccc::io {

/// Populate a @c traccc::silicon_detector_description object from text files.
///
/// @param dd The detector description object to set up.
/// @param geometry_file The path to the geometry description file.
/// @param digitization_file The path to the digitization configuration file.
/// @param geometry_format The format of the geometry description file.
/// @param digitization_format The format of the digitization configuration
///                            file.
///
void read_detector_description(
    silicon_detector_description::host& dd, std::string_view geometry_file,
    std::string_view digitization_file,
    data_format geometry_format = data_format::json,
    data_format digitization_format = data_format::json);

}  // namespace traccc::io
