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

/// @brief Compute each edge's slot in its destination node's incoming-edge
/// list.
///
/// One thread per edge atomically increments the per-destination-node count
/// in d_num_outgoing_edges_view and records the returned slot in
/// d_edge_links_view.  After this kernel the count buffer has been turned
/// into a write cursor that graph_edge_matching can read sequentially.
///
/// @param[in]  globalIndex                   Current thread index
/// @param[in]  d_edge_nodes_view             (src, dst) per edge
/// @param[out] d_edge_links_view             Per-edge slot in dst's list
/// @param[in,out] d_num_outgoing_edges_view  Per-node prefix-sum write cursor
///
TRACCC_HOST_DEVICE
inline void graph_edge_linking(
    const global_index_t globalIndex,
    const collection_types<uint2>::const_view& d_edge_nodes_view,
    const collection_types<unsigned int>::view d_edge_links_view,
    const collection_types<unsigned int>::view d_num_outgoing_edges_view);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/graph_edge_linking.ipp"
