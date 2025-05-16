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

TRACCC_DEVICE inline void add_block_offset(const global_index_t globalIndex,
                                     const add_block_offset_payload& payload) {

    vecmem::device_vector<const int> block_offsets(payload.block_offsets_view);
    vecmem::device_vector<int> prefix_sums(payload.prefix_sums_view);

    const unsigned int n_accepted = (*payload.update_res).n_accepted;

    if (globalIndex >= n_accepted || blockIdx.x == 0) {
        return;
    }

    prefix_sums[globalIndex] = block_offsets[blockIdx.x - 1];
}

}  // namespace traccc::device