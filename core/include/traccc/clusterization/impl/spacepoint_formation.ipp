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

TRACCC_HOST_DEVICE inline void fill_spacepoint(spacepoint& sp,
                                               const measurement& meas,
                                               const cell_module& module) {

    // Transform measurement position to 3D
    point3 local_3d = {meas.local[0], meas.local[1], 0.f};
    point3 global = module.placement.point_to_global(local_3d);

    // Fill spacepoint with this spacepoint
    sp = {global, meas};
}

}  // namespace traccc::details
