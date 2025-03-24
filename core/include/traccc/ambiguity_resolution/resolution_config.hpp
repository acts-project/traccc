/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <cstdint>

namespace traccc {

/// Configuration struct for ambiguity resolution
struct resolution_config {

    /// Minimum number of measurement to form a track.
    unsigned int min_meas_per_track = 3;

    /// Minimum number of shared measurement to be sent for competition with
    /// other tracks.
    unsigned int min_shared_meas_for_competition = 1;
};

}  // namespace traccc
