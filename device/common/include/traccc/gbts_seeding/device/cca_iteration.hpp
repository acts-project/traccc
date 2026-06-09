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
#include "traccc/device/concepts/barrier.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// System include(s).
#include <cstdint>

namespace traccc::device {

/// @brief One iteration of the cellular-automaton "longest path" relaxation.
///
/// Threads cooperatively process the current active-edge list, propagate
/// levels along the compact graph, and write the next iteration's active set
/// into the opposite ping-pong buffer (selected by iter parity).  The
/// final block to finish records the longest outgoing path summary per edge.
///
/// @param[in]  globalIndex               Edge index processed by this call
/// @param[in]  d_output_graph_view       Compact graph from graph_compression
/// @param[in,out] d_levels_view          Per-edge level ping-pong
/// @param[in,out] d_active_edges_view    Per-iteration active-edge list
/// @param[out] d_outgoing_paths_view     Longest outgoing path summary / edge
/// @param[in]  iter                      Iteration index
/// @param[in]  nConnectedEdges           Number of edges in the compact graph
/// @param[in]  max_num_neighbours        Maximum neighbours per edge
/// @param[in]  minLevel                  Minimum path length to remain active
///
TRACCC_HOST_DEVICE inline void cca_iteration(
    const global_index_t globalIndex,
    const collection_types<unsigned int>::const_view& d_output_graph_view,
    const collection_types<unsigned char>::view d_levels_view,
    const collection_types<char>::view d_active_edges_view,
    const collection_types<short2>::view d_outgoing_paths_view,
    const unsigned char iter, const unsigned int nConnectedEdges,
    const unsigned int max_num_neighbours, const unsigned char minLevel);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/cca_iteration.ipp"
