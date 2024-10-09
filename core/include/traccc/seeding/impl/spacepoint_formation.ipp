/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"

// Detray include(s).
#include "detray/geometry/tracking_surface.hpp"

namespace traccc::details {

TRACCC_HOST_DEVICE inline bool is_valid_measurement(const measurement& meas) {
    // We use 2D (pixel) measurements only for spacepoint creation
    if (meas.meas_dim == 2u) {
        return true;
    }
    return false;
}

template <typename detector_t>
TRACCC_HOST_DEVICE inline spacepoint create_spacepoint(
    const detector_t& det, const measurement& meas) {

    const detray::tracking_surface sf{det, meas.surface_link};

    // This local to global transformation only works for 2D planar
    // measurement
    // (e.g. barrel pixel and endcap pixel detector)
    const auto global = sf.bound_to_global({}, meas.local, {});

    // Return the spacepoint with this spacepoint
    return spacepoint{global, meas};
}

}  // namespace traccc::details