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
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void reset_edge_bids(
    const global_index_t globalIndex,
    const collection_types<int2>::const_view& d_path_store_view,
    const collection_types<int2>::view d_seed_proposals_view,
    const collection_types<unsigned long long int>::view d_edge_bids_view,
    const collection_types<char>::view d_seed_ambiguity_view,
    unsigned int& nRejectedPropsCounter) {

    const collection_types<int2>::const_device d_path_store(d_path_store_view);
    collection_types<int2>::device d_seed_proposals(d_seed_proposals_view);
    const collection_types<unsigned long long int>::const_device d_edge_bids(
        d_edge_bids_view);
    collection_types<char>::device d_seed_ambiguity(d_seed_ambiguity_view);

    // One proposal per call; the grid-stride loop lives in the kernel wrapper.
    const unsigned int prop_idx = globalIndex;

    const char ambi = d_seed_ambiguity[prop_idx];
    if ((ambi == -2) | (ambi == 0)) {
        // only reset maybes
        return;
    }
    const int2 prop = d_seed_proposals[prop_idx];

    bool isgood = true;
    // dummy path to start the loop
    int2 path = make_int2(0, prop.y);
    while (path.y >= 0) {
        path = d_path_store[static_cast<unsigned int>(path.y)];
        const unsigned long long int best_bid =
            d_edge_bids[static_cast<unsigned int>(path.x)];
        if (d_seed_ambiguity[static_cast<unsigned int>(best_bid &
                                                       0xFFFFFFFFLL)] == 0) {
            isgood = false;
            break;
        }
    }
    if (isgood) {
        // flag as maybe seed
        d_seed_ambiguity[prop_idx] = 1;
    } else {
        // definite fake
        d_seed_ambiguity[prop_idx] = -2;
        vecmem::device_atomic_ref<unsigned int>(nRejectedPropsCounter)
            .fetch_add(1u);
    }
}

}  // namespace traccc::device
