/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_parameters.hpp"

// Detray include(s).
#include "detray/geometry/barcode.hpp"

namespace traccc {

/// Track candidate contains the measurement object and its surface link
struct track_candidate {
    detray::geometry::barcode surface_link;
    measurement meas{};
};

/// Equality operator for track_candidate
TRACCC_HOST_DEVICE
inline bool operator==(const track_candidate& lhs, const track_candidate& rhs) {

    return ((lhs.surface_link == rhs.surface_link) && (lhs.meas == rhs.meas));
}

/// Declare a track candidates collection types
using track_candidate_collection_types = collection_types<track_candidate>;
/// Declare a track candidates container type
using track_candidate_container_types =
    container_types<bound_track_parameters, track_candidate>;

}  // namespace traccc