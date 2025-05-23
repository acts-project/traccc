/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Traccc include(s).
#include "traccc/device/concepts/barrier.hpp"

namespace traccc::device {

TRACCC_DEVICE inline void reset_status(const global_index_t globalIndex,
                                       const reset_status_payload& payload) {

    if (*(payload.is_first_iteration) == 1) {
        *(payload.terminate) = 0;
        *(payload.is_first_iteration) = 0;
    } else {
        if (*(payload.max_shared) == 0) {
            *(payload.terminate) = 1;
        }
    }

    if (*(payload.terminate) == 0) {
        *(payload.max_shared) = 0;
        *(payload.n_updated_tracks) = 0;
    }
}

}  // namespace traccc::device
