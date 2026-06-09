/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/global_index.hpp"

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

namespace traccc::device {

/// @brief Pack kept edges into the compact "output graph" layout.
///
/// Each thread processes one original edge; if it survived re-indexing, the
/// thread writes a record at its compact slot containing the source/destination
/// original-SP indices, the neighbour count, and up to nMaxNei remapped
/// neighbour indices.
///
/// @param[in]  globalIndex                  Current thread index
/// @param[in]  d_orig_node_index_view       Sorted slot → original SP index
/// @param[in]  d_edge_nodes_view            (src, dst) per original edge
/// @param[in]  d_num_neighbours_view        Accepted neighbour count per edge
/// @param[in]  d_neighbours_view            Neighbour edge indices per edge
/// @param[in]  d_reIndexer_view             Old-edge → compact-edge map
/// @param[out] d_output_graph_view          Compact graph in row-major layout:
///                                          each edge owns a contiguous block
///                                          of edge_size = 2 + 1 + nMaxNei ints
///                                          ([node1, node2, nNei,
///                                          nei0..neiN-1]).
/// @param[in]  nMaxNei                      Maximum neighbours per edge
///
TRACCC_HOST_DEVICE
inline void graph_compression(
    const global_index_t globalIndex,
    const collection_types<unsigned int>::const_view& d_orig_node_index_view,
    const collection_types<uint2>::const_view& d_edge_nodes_view,
    const collection_types<unsigned char>::const_view& d_num_neighbours_view,
    const collection_types<unsigned int>::const_view& d_neighbours_view,
    const collection_types<int>::const_view& d_reIndexer_view,
    const collection_types<unsigned int>::view& d_output_graph_view,
    const unsigned int nMaxNei);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/graph_compression.ipp"
