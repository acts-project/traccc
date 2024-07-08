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
#include "traccc/geometry/geometry.hpp"

// Detray include(s).
#include <detray/geometry/barcode.hpp>
#include <detray/geometry/tracking_surface.hpp>

// System include(s).
#include <cstdint>
#include <map>
#include <memory>
#include <string_view>
#include <utility>

namespace traccc::io {

/// Read in the detector geometry description from an input file
///
/// @param filename The name of the input file to read
/// @param format The format of the input file
/// @return A description of the detector modules
///
std::pair<geometry,
          std::unique_ptr<std::map<std::uint64_t, detray::geometry::barcode>>>
read_geometry(std::string_view filename, data_format format = data_format::csv);

/// Read in the detector geometry description from a detector object
template <typename detector_t>
geometry alt_read_geometry(const detector_t& det) {

    std::map<traccc::geometry_id, traccc::transform3> maps;
    using cxt_t = typename detector_t::geometry_context;
    const cxt_t ctx0{};

    for (const auto& sf_desc : det.surfaces()) {
        const detray::tracking_surface sf{det, sf_desc.barcode()};

        maps.insert({sf.barcode().value(), sf.transform(ctx0)});
    }

    return maps;
}

}  // namespace traccc::io
