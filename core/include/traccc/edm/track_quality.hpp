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
    void reset_quality() {
        ndf = 0.f;
        chi2 = 0.f;
        n_holes = 0u;
    }
};

/// Equality operator for track quality
TRACCC_HOST_DEVICE
inline bool operator==(const track_quality& lhs, const track_quality& rhs) {

    return ((math::fabs(lhs.ndf - rhs.ndf) < float_epsilon) &&
            (math::fabs(lhs.chi2 - rhs.chi2) < float_epsilon) &&
            (lhs.n_holes == rhs.n_holes));
}

}  // namespace traccc
