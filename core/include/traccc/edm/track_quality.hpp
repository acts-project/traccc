/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"

namespace traccc {

/// Track quality
struct track_quality {

    /// Number of degree of freedoms of fitted track
    traccc::scalar ndf{0};

    /// Chi square of fitted track
    traccc::scalar chi2{0};

    // The number of holes (The number of sensitive surfaces which do not have a
    // measurement for the track pattern)
    unsigned int n_holes{0u};

    /// Reset the summary
    TRACCC_HOST_DEVICE
    void reset() {
        ndf = 0.f;
        chi2 = 0.f;
        n_holes = 0u;
    }
};

}  // namespace traccc