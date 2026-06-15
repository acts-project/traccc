/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/thread_id.hpp"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/memory/device_atomic_ref.hpp>

// System include(s).
#include <cstdint>

namespace traccc::device {

template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_reindex_edges(
    const thread_id_t& thread_id, const gbts_reindex_edges_payload& payload) {

    vecmem::device_vector<int> d_reIndexer(payload.reIndexer);

    for (unsigned int globalIndex = thread_id.getGlobalThreadIdX();
         globalIndex < payload.nEdges;
         globalIndex += thread_id.getBlockDimX() * thread_id.getGridDimX()) {

        if (d_reIndexer[globalIndex] == -1) {
            continue;
        }
        d_reIndexer[globalIndex] =
            static_cast<int>(vecmem::device_atomic_ref<unsigned int>(
                                 *payload.nConnectedEdgesCounter)
                                 .fetch_add(1u));
    }
}

}  // namespace traccc::device
