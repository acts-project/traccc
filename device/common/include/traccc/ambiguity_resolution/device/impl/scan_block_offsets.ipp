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
TRACCC_DEVICE inline void scan_block_offsets(
    const global_index_t globalIndex, const unsigned int blockSize,
    const unsigned int threadIndex, const barrier_t& barrier,
    const scan_block_offsets_payload& payload, int* shared_temp) {

    if (*(payload.terminate) == 1 || *(payload.n_updated_tracks) == 0) {
        return;
    }

    vecmem::device_vector<const int> block_offsets(payload.block_offsets_view);
    vecmem::device_vector<int> scanned_block_offsets(
        payload.scanned_block_offsets_view);

    int n_blocks = (*(payload.n_accepted) + 1023) / 1024;

    barrier.blockBarrier();

    // 1. Load from global to shared
    int value = 0;
    if (threadIndex < n_blocks) {
        value = block_offsets[threadIndex];
    }
    shared_temp[threadIndex] = value;
    barrier.blockBarrier();

    // 2. Inclusive scan (Hillis-Steele style)
    for (int offset = 1; offset < n_blocks; offset *= 2) {
        int temp = 0;
        if (threadIndex >= offset) {
            temp = shared_temp[threadIndex - offset];
        }
        barrier.blockBarrier();
        shared_temp[threadIndex] += temp;
        barrier.blockBarrier();
    }

    // 3. Write back
    if (threadIndex < n_blocks) {
        scanned_block_offsets[threadIndex] = shared_temp[threadIndex];
    }
}

}  // namespace traccc::device
