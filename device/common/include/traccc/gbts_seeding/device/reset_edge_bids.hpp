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

/// @brief Mark a losing seed proposal against the current edge bids.
///
/// Processes one proposal (the grid-stride loop lives in the kernel wrapper):
/// compares it against the winning bid recorded in d_edge_bids_view, and either
/// updates d_seed_ambiguity_view or atomically increments nRejectedPropsCounter
/// if it loses.
///
/// @param[in]  globalIndex                 Proposal index processed by this
/// call
/// @param[in]  d_path_store_view           Per-path (parent, edge) entries
/// @param[in,out] d_seed_proposals_view    Per-seed (path index, level)
/// @param[in,out] d_edge_bids_view         Per-edge highest-bidder seed
/// @param[in,out] d_seed_ambiguity_view    Per-seed ambiguity tag
/// @param[in,out] nRejectedPropsCounter    Global atomic rejected-count
///
TRACCC_HOST_DEVICE
inline void reset_edge_bids(
    const global_index_t globalIndex,
    const collection_types<int2>::const_view& d_path_store_view,
    const collection_types<int2>::view d_seed_proposals_view,
    const collection_types<unsigned long long int>::view d_edge_bids_view,
    const collection_types<char>::view d_seed_ambiguity_view,
    unsigned int& nRejectedPropsCounter);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/reset_edge_bids.ipp"
