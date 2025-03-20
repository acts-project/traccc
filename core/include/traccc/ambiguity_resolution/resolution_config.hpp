/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"

namespace traccc {

/// Configuration struct for ambiguity resolution
struct resolution_config {

    /// Minimum number of measurement to form a track.
    unsigned int min_measurements_per_track = 3;

    /// Maximum amount of shared hits per track. One (1) means "no shared
    /// hit allowed".
    unsigned int maximum_shared_hits = 1;

    /// Maximum number of iterations.
    unsigned int maximum_iterations = 1000000;
};

}  // namespace traccc
