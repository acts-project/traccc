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
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void graph_edge_linking(
    const global_index_t globalIndex,
    const collection_types<uint2>::const_view& d_edge_nodes_view,
    const collection_types<unsigned int>::view d_edge_links_view,
    const collection_types<unsigned int>::view d_num_outgoing_edges_view) {

    const collection_types<uint2>::const_device d_edge_nodes(d_edge_nodes_view);
    collection_types<unsigned int>::device d_edge_links(d_edge_links_view);
    collection_types<unsigned int>::device d_num_outgoing_edges(
        d_num_outgoing_edges_view);

    const unsigned int sharedNode = d_edge_nodes[globalIndex].y;
    const unsigned int pos = vecmem::device_atomic_ref<unsigned int>(
                                 d_num_outgoing_edges[sharedNode])
                                 .fetch_sub(1u);
    d_edge_links[pos - 1u] = static_cast<unsigned int>(globalIndex);
}

}  // namespace traccc::device
