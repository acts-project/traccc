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
    const block_inclusive_scan_payload& payload) {}

}  // namespace traccc::device