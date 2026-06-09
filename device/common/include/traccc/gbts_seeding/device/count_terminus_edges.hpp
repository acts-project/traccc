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

/// @brief Count terminus edges (those with no live outgoing path) and total
/// paths.
///
/// Each block scans a chunk of edges, accumulates a per-block tally of
/// terminus edges into shared outgoingCount, then atomically folds the
/// tally into nPathStoreSizeCounter and the total reachable-paths sum into
/// nPathsCounter.
///
/// @param[in]  blockIndex                  CUDA block index
/// @param[in]  threadIndex                 CUDA thread index in block
/// @param[in]  blockSize                   CUDA block size
/// @param[in]  barrier                     Block-wide barrier
/// @param[in,out] outgoingCount            Shared-mem per-block tally
/// @param[in]  d_outgoing_paths_view       Per-edge longest-path summary
/// @param[in,out] nPathsCounter            Total reachable paths
/// @param[in,out] nPathStoreSizeCounter    Running terminus-edge count
///
TRACCC_HOST_DEVICE inline void count_terminus_edges(
    const global_index_t globalIndex,
    const collection_types<short2>::view d_outgoing_paths_view,
    unsigned int& nPathsCounter, unsigned int& nPathStoreSizeCounter);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/count_terminus_edges.ipp"
