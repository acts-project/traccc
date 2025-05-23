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

template <concepts::barrier barrier_t>
TRACCC_DEVICE inline void block_inclusive_scan(
    const global_index_t globalIndex, const unsigned int blockSize,
    const unsigned int blockIndex, const unsigned int threadIndex,
    const barrier_t& barrier, const block_inclusive_scan_payload& payload,
    int* shared_temp) {

    if (*(payload.terminate) == 1 || *(payload.n_updated_tracks) == 0) {
        return;
    }

    vecmem::device_vector<const unsigned int> sorted_ids(
        payload.sorted_ids_view);
    vecmem::device_vector<const int> is_updated(payload.is_updated_view);
    vecmem::device_vector<int> block_offsets(payload.block_offsets_view);
    vecmem::device_vector<int> prefix_sums(payload.prefix_sums_view);

    const unsigned int n_accepted = *(payload.n_accepted);

    if (globalIndex >= n_accepted) {
        shared_temp[threadIndex] = 0;
    } else {
        shared_temp[threadIndex] = is_updated[sorted_ids[globalIndex]];
    }

    barrier.blockBarrier();

    // inclusive scan in shared memory
    for (int stride = 1; stride < blockSize; stride *= 2) {
        int val = 0;
        if (threadIndex >= stride) {
            val = shared_temp[threadIndex - stride];
        }
        barrier.blockBarrier();
        if (threadIndex >= stride) {
            shared_temp[threadIndex] += val;
        }
        barrier.blockBarrier();
    }

    if (globalIndex < n_accepted) {
        prefix_sums[globalIndex] = shared_temp[threadIndex];
    }

    barrier.blockBarrier();

    if (threadIndex == blockSize - 1) {
        block_offsets[blockIndex] = shared_temp[threadIndex];
    }
}

}  // namespace traccc::device
