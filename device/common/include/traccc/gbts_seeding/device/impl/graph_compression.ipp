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
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void graph_compression(
    const global_index_t globalIndex,
    const collection_types<unsigned int>::const_view& d_orig_node_index_view,
    const collection_types<uint2>::const_view& d_edge_nodes_view,
    const collection_types<unsigned char>::const_view& d_num_neighbours_view,
    const collection_types<unsigned int>::const_view& d_neighbours_view,
    const collection_types<int>::const_view& d_reIndexer_view,
    const collection_types<unsigned int>::view& d_output_graph_view,
    const unsigned int nMaxNei) {

    const collection_types<unsigned int>::const_device d_orig_node_index(
        d_orig_node_index_view);
    const collection_types<uint2>::const_device d_edge_nodes(d_edge_nodes_view);
    const collection_types<unsigned char>::const_device d_num_neighbours(
        d_num_neighbours_view);
    const collection_types<unsigned int>::const_device d_neighbours(
        d_neighbours_view);
    const collection_types<int>::const_device d_reIndexer(d_reIndexer_view);
    collection_types<unsigned int>::device d_output_graph(d_output_graph_view);

    const int newIdx = d_reIndexer[globalIndex];
    if (newIdx == -1) {
        return;
    }

    // Row-major output graph: each edge owns a contiguous block of
    // edge_size = 2 + 1 + nMaxNei ints ([node1, node2, nNei, nei0..neiN-1]).
    const unsigned int edge_size = 2u + 1u + nMaxNei;
    const unsigned int pos = edge_size * static_cast<unsigned int>(newIdx);

    const uint2 edge_nodes = d_edge_nodes[globalIndex];
    d_output_graph[pos + gbts_consts::node1] = d_orig_node_index[edge_nodes.x];
    d_output_graph[pos + gbts_consts::node2] = d_orig_node_index[edge_nodes.y];

    const unsigned char nNei = d_num_neighbours[globalIndex];
    d_output_graph[pos + gbts_consts::nNei] = nNei;
    const unsigned int nei_pos = nMaxNei * globalIndex;
    for (unsigned int k = 0u; k < nNei; k++) {
        d_output_graph[pos + gbts_consts::nei_start + k] =
            d_reIndexer[d_neighbours[nei_pos + k]];
    }
}

}  // namespace traccc::device
