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

/// @brief Seed the path store with one entry per terminus edge.
///
/// Each thread inspects one edge; if it is a terminus (no live outgoing path
/// in d_outgoing_paths_view), it writes a (parent = -1, edge) record into
/// the path store, marking the root of a future path traversal.
///
/// @param[in]  globalIndex               Current thread index
/// @param[out] d_path_store_view         Path-store entries (terminus prefix)
/// @param[in]  d_outgoing_paths_view     Per-edge longest-path summary
///
TRACCC_HOST_DEVICE
inline void add_terminus_to_path_store(
    const global_index_t globalIndex,
    const collection_types<int2>::view d_path_store_view,
    const collection_types<short2>::const_view& d_outgoing_paths_view);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/add_terminus_to_path_store.ipp"
