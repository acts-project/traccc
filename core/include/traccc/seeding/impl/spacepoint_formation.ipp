/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"

// Detray include(s).
#include <detray/geometry/tracking_surface.hpp>

namespace traccc::details {

TRACCC_HOST_DEVICE inline bool is_valid_measurement(const measurement& meas) {
    // We use 2D (pixel) measurements only for spacepoint creation
    return (meas.meas_dim == 2u);
}

template <typename soa_t, typename detector_t>
TRACCC_HOST_DEVICE inline void fill_pixel_spacepoint(
    edm::spacepoint<soa_t>& sp, const detector_t& det, const measurement& meas,
    const typename detector_t::geometry_context gctx) {

    // Get the global position of this silicon pixel measurement.
    const detray::tracking_surface sf{det, meas.surface_link};
    const auto global = sf.local_to_global(gctx, meas.local, {});

    // Fill the spacepoint with the global position and the measurement.
    sp.x() = global[0];
    sp.y() = global[1];
    sp.z() = global[2];
    sp.radius_variance() = 0.f;
    sp.z_variance() = 0.f;
}

}  // namespace traccc::details
