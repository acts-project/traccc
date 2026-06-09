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

/// @brief For each edge, find compatible neighbour edges sharing its outer
/// node.
///
/// One thread per edge pair-tests the edge against every edge leaving its outer
/// node using the packed edge parameters, recording up to nMaxNei accepted
/// neighbours, marking the edge as "kept", and atomically incrementing
/// nConnectionsCounter.
///
/// @param[in]  globalIndex                  Global thread index (one per edge)
/// @param[in]  d_graph_matching_params      Pair-matching cuts
/// @param[in]  d_edge_params_view           Per-edge [exp_eta, curv, phi_z,
/// phi_w]
/// @param[in]  d_edge_nodes_view            (src, dst) per edge
/// @param[in]  d_num_outgoing_edges_view    Per-node prefix sum (now offsets)
/// @param[in]  d_edge_links_view            Per-edge slot in its node's list
/// @param[out] d_num_neighbours_view        Accepted neighbour count per edge
/// @param[out] d_neighbours_view            Neighbour edge indices per edge
/// @param[out] d_reIndexer_view             Per-edge "kept" flag
/// @param[in,out] nConnectionsCounter       Global connection-count atomic
/// @param[in]  nMaxNei                      Max neighbours retained per edge
///
TRACCC_HOST_DEVICE inline void graph_edge_matching(
    const global_index_t globalIndex,
    const gbts_edge_matching_params& d_graph_matching_params,
    const collection_types<gbts_edge4>::const_view& d_edge_params_view,
    const collection_types<uint2>::const_view& d_edge_nodes_view,
    const collection_types<unsigned int>::const_view& d_num_outgoing_edges_view,
    const collection_types<unsigned int>::const_view& d_edge_links_view,
    const collection_types<unsigned char>::view& d_num_neighbours_view,
    const collection_types<unsigned int>::view& d_neighbours_view,
    const collection_types<int>::view& d_reIndexer_view,
    unsigned int& nConnectionsCounter, const unsigned int nMaxNei);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/graph_edge_matching.ipp"
