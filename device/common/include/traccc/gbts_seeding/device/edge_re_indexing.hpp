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

namespace traccc::device {

/// @brief Replace the per-edge "kept" flag with its compacted index.
///
/// Each thread reads its slot in d_reIndexer_view; if the edge is marked
/// "kept", it atomically claims the next slot in nConnectedEdges and
/// writes that compact index back; otherwise the slot is set to a sentinel.
///
/// @param[in]  globalIndex            Current thread index
/// @param[in,out] d_reIndexer_view    Per-edge "kept" in, compact index out
/// @param[in,out] nConnectedEdges     Atomic counter of surviving edges
///
TRACCC_HOST_DEVICE
inline void edge_re_indexing(const global_index_t globalIndex,
                             const collection_types<int>::view d_reIndexer_view,
                             unsigned int& nConnectedEdges);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/edge_re_indexing.ipp"
