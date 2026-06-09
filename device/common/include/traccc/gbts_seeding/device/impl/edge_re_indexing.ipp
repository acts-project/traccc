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
#include "traccc/edm/container.hpp"

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

// System include(s).
#include <cstdint>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void edge_re_indexing(const global_index_t globalIndex,
                             const collection_types<int>::view d_reIndexer_view,
                             unsigned int& nConnectedEdges) {

    collection_types<int>::device d_reIndexer(d_reIndexer_view);

    if (d_reIndexer[globalIndex] == -1) {
        return;
    }
    d_reIndexer[globalIndex] = static_cast<int>(
        vecmem::device_atomic_ref<unsigned int>(nConnectedEdges).fetch_add(1u));
}

}  // namespace traccc::device
