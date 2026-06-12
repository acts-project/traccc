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
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

// System include(s).
#include <cstdint>

namespace traccc::device {

/// (Global Event Data) Payload for the @c
/// traccc::device::gbts_rebid_seeds_for_edges function
struct gbts_rebid_seeds_for_edges_payload {
    /// Number of seed proposals
    unsigned int nProps;
    /// Per-path (edge index, parent path-store index or -1) entries
    vecmem::data::vector_view<const int2> path_store;
    /// Per-seed-proposal (path_store index, level)
    vecmem::data::vector_view<int2> seed_proposals;
    /// In/out: per-edge highest-bidder seed proposal (cleared on entry)
    vecmem::data::vector_view<unsigned long long int> edge_bids;
    /// In/out: per-seed-proposal ambiguity tag
    vecmem::data::vector_view<char> seed_ambiguity;
    /// In/out: global atomic counter of rejected proposals
    unsigned int* nRejectedPropsCounter;
    /// True on the first bidding round (folds the init pass)
    bool first_round;
};

/// @brief Have one surviving seed re-bid for every edge along its path.
///
/// Processes one proposal (the grid-stride loop lives in the kernel wrapper):
/// for every edge on its path-store chain it atomically compares its packed
/// bid against the edge bids and replaces the entry if the new bid wins.
///
/// @param[in] globalIndex Proposal index processed by this call
/// @param[in] payload     The global memory payload
///
TRACCC_HOST_DEVICE
inline void gbts_rebid_seeds_for_edges(
    const global_index_t globalIndex,
    const gbts_rebid_seeds_for_edges_payload& payload);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/gbts_rebid_seeds_for_edges.ipp"
