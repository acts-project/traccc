/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"

namespace traccc::details {

TRACCC_HOST_DEVICE inline void fill_spacepoint(
    spacepoint& sp, const measurement& meas,
    const silicon_detector_description::const_device& dd) {

    // Transform measurement position to 3D
    const point3 local_3d = {meas.local[0], meas.local[1], 0.f};
    sp.global = dd.placement().at(meas.module_link).point_to_global(local_3d);
    sp.meas = meas;
}

}  // namespace traccc::details
