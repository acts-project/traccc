/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/barrier.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

namespace traccc::device {

/// @brief Walk each terminus edge backwards along live levels, growing the path
/// store.
///
/// One block expands nTerminusPerBlock terminus seeds at once using
/// a shared "live paths" frontier.  At each step the block reads neighbour
/// edges that match the next-lower level, atomically reserves slots in the
/// path store via nPathStoreSizeCounter, and continues until all paths
/// reach the graph boundary or until the frontier is empty.
///
/// @param[in]  thread_id                 Thread/block identifier (one
/// block/task)
/// @param[in]  barrier                   Block-wide barrier
/// @param[in,out] live_paths             Shared-mem frontier of in-flight paths
/// @param[in,out] n_live_paths           Shared-mem frontier size
/// @param[out] d_path_store_view         Path-store entries
/// @param[in]  d_output_graph_view       Compact graph from graph_compression
/// @param[in]  d_levels_view             Per-edge CCA level array
/// @param[in,out] nPathStoreSizeCounter  Global atomic write cursor
/// @param[in]  nTerminus                 Number of terminus edges
/// @param[in]  nTerminusPerBlock         Terminus edges processed per block
/// @param[in]  max_num_neighbours        Maximum neighbours per edge
/// @param[in]  nPaths                    Upper bound on the number of paths
///
/// Shared-memory scratch for fill_path_store: the block-local stack of live
/// paths being walked and its running length.
struct fill_path_store_shared_payload {
    traccc::uint2* live_paths;
    int& n_live_paths;
};

template <concepts::thread_id1 thread_id_t, concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void fill_path_store(
    const thread_id_t& thread_id, const barrier_t& barrier,
    const fill_path_store_shared_payload& shared,
    collection_types<int2>::view d_path_store_view,
    const collection_types<unsigned int>::const_view& d_output_graph_view,
    const collection_types<unsigned char>::const_view& d_levels_view,
    unsigned int& nPathStoreSizeCounter, const unsigned int nTerminus,
    const unsigned int nTerminusPerBlock, const unsigned int max_num_neighbours,
    const unsigned int nPaths);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/fill_path_store.ipp"
