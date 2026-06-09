/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

// System include(s).
#include <cstdint>

namespace traccc::device {

TRACCC_HOST_DEVICE inline void count_terminus_edges(
    const global_index_t globalIndex,
    const collection_types<short2>::view d_outgoing_paths_view,
    unsigned int& nPathsCounter, unsigned int& nPathStoreSizeCounter) {

    collection_types<short2>::device d_outgoing_paths(d_outgoing_paths_view);

    // One edge per call; the grid-stride loop lives in the kernel wrapper.
    const short2 out_paths = d_outgoing_paths[globalIndex];
    if (out_paths.y != -1) {
        d_outgoing_paths[globalIndex].y = static_cast<short>(
            vecmem::device_atomic_ref<unsigned int>(nPathStoreSizeCounter)
                .fetch_add(1u));
        vecmem::device_atomic_ref<unsigned int>(nPathsCounter)
            .fetch_add(static_cast<unsigned int>(out_paths.x));
    }
}

}  // namespace traccc::device
