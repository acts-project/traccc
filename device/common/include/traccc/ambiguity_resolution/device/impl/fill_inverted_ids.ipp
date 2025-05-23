/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Traccc include(s).
#include "traccc/device/concepts/barrier.hpp"

// Thrust include(s).
#include <thrust/execution_policy.h>
#include <thrust/find.h>

namespace traccc::device {

TRACCC_DEVICE inline void fill_inverted_ids(
    const global_index_t globalIndex,
    const fill_inverted_ids_payload& payload) {

    if (*(payload.terminate) == 1 || *(payload.n_updated_tracks) == 0) {
        return;
    }

    vecmem::device_vector<const unsigned int> sorted_ids(
        payload.sorted_ids_view);
    vecmem::device_vector<unsigned int> inverted_ids(payload.inverted_ids_view);

    const unsigned int n_accepted = *(payload.n_accepted);

    if (globalIndex >= n_accepted) {
        return;
    }

    inverted_ids[sorted_ids[globalIndex]] = globalIndex;
}

}  // namespace traccc::device
