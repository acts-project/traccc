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

namespace traccc {

/// Track candidate contains the measurement object and its surface link
struct track_candidate {
    geometry_id surface_link;
    measurement meas;
};

/// Declare a track candidates container type
using track_candidate_container_types =
    container_types<bound_track_parameters, track_candidate>;

}  // namespace traccc