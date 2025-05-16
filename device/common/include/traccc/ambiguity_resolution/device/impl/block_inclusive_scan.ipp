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
    const global_index_t globalIndex, const barrier_t& barrier,
    const block_inclusive_scan_payload& payload) {

    vecmem::device_vector<const unsigned int> sorted_ids(payload.sorted_ids_view);
    vecmem::device_vector<const int> is_updated(payload.is_updated_view);
    vecmem::device_vector<int> block_offsets(payload.block_offsets_view);
    vecmem::device_vector<int> prefix_sums(payload.prefix_sums_view);

    __shared__ int temp[32];

    const unsigned int n_accepted = (*payload.update_res).n_accepted;

    if (globalIndex >= n_accepted) {
        temp[threadIdx.x] = 0;
    } else {
        temp[threadIdx.x] = is_updated[sorted_ids[globalIndex]];
    }
    __syncthreads();

    // inclusive scan in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int val = 0;
        if (threadIdx.x >= stride) {
            val = temp[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride) {
            temp[threadIdx.x] += val;
        }
        __syncthreads();
    }

    if (globalIndex < n_accepted) {
        prefix_sums[globalIndex] = temp[threadIdx.x];
    }

    if (threadIdx.x == blockDim.x - 1) {
        block_offsets[blockIdx.x] = temp[threadIdx.x];
    }
}

}  // namespace traccc::device