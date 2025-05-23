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

TRACCC_DEVICE inline void gather_tracks(const global_index_t globalIndex,
                                        const gather_tracks_payload& payload) {

    if (*(payload.terminate) == 1 || *(payload.n_updated_tracks) == 0) {
        return;
    }

    vecmem::device_vector<const unsigned int> temp_sorted_ids(
        payload.temp_sorted_ids_view);
    vecmem::device_vector<unsigned int> sorted_ids(payload.sorted_ids_view);
    vecmem::device_vector<int> is_updated(payload.is_updated_view);

    const unsigned int n_accepted = *(payload.n_accepted);

    // Reset is_updated vector
    if (globalIndex < is_updated.size()) {
        is_updated[globalIndex] = 0;
    }

    if (globalIndex >= n_accepted) {
        return;
    }

    sorted_ids[globalIndex] = temp_sorted_ids[globalIndex];
}

}  // namespace traccc::device
