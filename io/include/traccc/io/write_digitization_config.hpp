/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/io/data_format.hpp"

// Project include(s).
#include "traccc/io/digitization_config.hpp"

// Detray include(s).
#include "detray/geometry/barcode.hpp"

// System include(s).
#include <string_view>
#include <utility>
#include <vector>

namespace Acts {
class GeometryIdentifier;
}

namespace traccc::io {

using digitization_out_collection = std::vector<
    std::pair<detray::geometry::barcode, module_digitization_config>>;

/// Write the detector digitization configuration to an output file
///
/// @param filename The name of the file to write the data to
/// @param digi_cfg The digitization to be written to file
/// @param format The format of the output file
///
void write_digitization_config(std::string_view filename,
                               const digitization_out_collection &digi_cfg,
                               data_format format = data_format::json);

}  // namespace traccc::io
