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

// Detray include(s).
#include "detray/geometry/surface.hpp"

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
    const auto surfaces = det.surface_lookup();
    std::map<traccc::geometry_id, traccc::transform3> maps;
    using cxt_t = typename detector_t::geometry_context;
    const cxt_t ctx0{};

    for (const auto& sf_desc : surfaces) {
        const detray::surface<detector_t> sf{det, sf_desc.barcode()};

        maps.insert({sf.barcode().value(), sf.transform(ctx0)});
    }

    return maps;
}

}  // namespace traccc::io
