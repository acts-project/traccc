/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/io/data_format.hpp"

// Project include(s).
#include "traccc/geometry/geometry.hpp"

// System include(s).
#include <string_view>

namespace traccc::io {

/// Read in the detector geometry description from an input file
///
/// @param filename The name of the input file to read
/// @param format The format of the input file
/// @return A description of the detector modules
///
geometry read_geometry(std::string_view filename,
                       data_format format = data_format::csv);

/// Read in the detector geometry description from a detector object
template <typename detector_t>
geometry alt_read_geometry(const detector_t& det) {

    const auto transforms = det.transform_store();
    const auto surfaces = det.surfaces();
    std::map<traccc::geometry_id, traccc::transform3> maps;

    for (const auto& sf : surfaces) {
        maps.insert({sf.barcode().value(), transforms[sf.transform()]});
    }

    return maps;
}

}  // namespace traccc::io
