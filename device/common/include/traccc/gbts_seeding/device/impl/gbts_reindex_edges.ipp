/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/global_index.hpp"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/memory/device_atomic_ref.hpp>

// System include(s).
#include <cstdint>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void gbts_reindex_edges(const global_index_t globalIndex,
                               const gbts_reindex_edges_payload& payload) {

    vecmem::device_vector<int> d_reIndexer(payload.reIndexer);

    if (d_reIndexer[globalIndex] == -1) {
        return;
    }
    d_reIndexer[globalIndex] = static_cast<int>(
        vecmem::device_atomic_ref<unsigned int>(*payload.nConnectedEdgesCounter)
            .fetch_add(1u));
}

}  // namespace traccc::device
