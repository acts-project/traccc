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

    vecmem::device_vector<const int>(payload.is_updated_view);
    vecmem::device_vector<int> block_offsets(payload.block_offsets_view);
    vecmem::device_vector<int> prefix_sums(payload.prefix_sums_view);
}

}  // namespace traccc::device