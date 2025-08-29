/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_candidate_collection.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/geometry/host_detector.hpp"

// System include(s).
#include <string_view>

namespace traccc::io::obj {

/// Write a track candidate container into a Wavefront OBJ file.
///
/// @param filename is the name of the output file
/// @param tracks is the track candidate container to write
/// @param measurements is the collection of all measurements in the event
/// @param detector is the Detray detector describing the geometry
///
void write_track_candidates(
    std::string_view filename,
    edm::track_candidate_collection<default_algebra>::const_view tracks,
    measurement_collection_types::const_view measurements,
    const traccc::host_detector& detector);

}  // namespace traccc::io::obj
