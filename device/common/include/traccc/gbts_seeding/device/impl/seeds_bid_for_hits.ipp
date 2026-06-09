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

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void seeds_bid_for_hits(
    const global_index_t globalIndex,
    const collection_types<unsigned int>::const_view& d_output_graph_view,
    const collection_types<int2>::const_view& d_seed_proposals_view,
    const collection_types<int2>::const_view& d_path_store_view,
    const collection_types<char>::const_view& d_seed_ambiguity_view,
    const collection_types<unsigned long long int>::view d_hit_bids_view,
    const unsigned int edge_size) {

    const collection_types<unsigned int>::const_device d_output_graph(
        d_output_graph_view);
    const collection_types<int2>::const_device d_path_store(d_path_store_view);
    const collection_types<char>::const_device d_seed_ambiguity(
        d_seed_ambiguity_view);
    const collection_types<int2>::const_device d_seed_proposals(
        d_seed_proposals_view);
    collection_types<unsigned long long int>::device d_hit_bids(
        d_hit_bids_view);

    // One proposal per call; the grid-stride loop lives in the kernel wrapper.
    const unsigned int prop_idx = globalIndex;
    if (d_seed_ambiguity[prop_idx] == -2) {
        return;
    }
    const int2 prop = d_seed_proposals[prop_idx];
    const unsigned long long int seed_bid =
        (static_cast<unsigned long long int>(prop.x) << 32) |
        (static_cast<unsigned long long int>(prop_idx));

    int2 path = make_int2(0, prop.y);
    while (path.y >= 0) {
        path = d_path_store[static_cast<unsigned int>(path.y)];
        const unsigned int sp_idx =
            d_output_graph[edge_size * static_cast<unsigned int>(path.x) +
                           gbts_consts::node1];
        vecmem::device_atomic_ref<unsigned long long int> atomic_bid(
            d_hit_bids[sp_idx]);
        atomic_bid.fetch_max(seed_bid);
    }
    const unsigned int sp_idx =
        d_output_graph[edge_size * static_cast<unsigned int>(path.x) +
                       gbts_consts::node2];
    vecmem::device_atomic_ref<unsigned long long int> atomic_bid(
        d_hit_bids[sp_idx]);
    atomic_bid.fetch_max(seed_bid);
}

}  // namespace traccc::device
