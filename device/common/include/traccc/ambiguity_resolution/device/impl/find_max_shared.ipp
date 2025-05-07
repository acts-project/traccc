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
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>

namespace traccc::device {

template <concepts::barrier barrier_t>
TRACCC_DEVICE inline void find_max_shared(
    const global_index_t globalIndex, const barrier_t& barrier,
    const find_max_shared_payload& payload) {

    // Unsafe for multiple blocks!
    if (globalIndex == 0) {
        (*payload.update_res).max_shared = 0;
    }

    vecmem::device_vector<const unsigned int> sorted_ids(
        payload.sorted_ids_view);
    vecmem::device_vector<const unsigned int> n_shared(payload.n_shared_view);

    if (globalIndex >= *payload.n_accepted) {
        return;
    }

    auto tid = sorted_ids[globalIndex];
    auto shared = n_shared[tid];

    for (int offset = 16; offset > 0; offset >>= 1) {
        unsigned int other_tid = __shfl_down_sync(0xffffffff, tid, offset);
        unsigned int other_shared =
            __shfl_down_sync(0xffffffff, shared, offset);

        if (other_shared > shared) {
            tid = other_tid;
            shared = other_shared;
        }
    }

    if (threadIdx.x == 0) {
        atomicMax(&((*payload.update_res).max_shared), shared);
    }
}

}  // namespace traccc::device
