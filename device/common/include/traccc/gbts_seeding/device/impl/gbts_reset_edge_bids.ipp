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
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/memory/device_atomic_ref.hpp>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void gbts_reset_edge_bids(const global_index_t globalIndex,
                                 const gbts_reset_edge_bids_payload& payload) {

    const vecmem::device_vector<const int2> d_path_store(payload.path_store);
    vecmem::device_vector<int2> d_seed_proposals(payload.seed_proposals);
    const vecmem::device_vector<const unsigned long long int> d_edge_bids(
        payload.edge_bids);
    vecmem::device_vector<char> d_seed_ambiguity(payload.seed_ambiguity);

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
    int2 path = int2{0, prop.y};
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
        vecmem::device_atomic_ref<unsigned int>(*payload.nRejectedPropsCounter)
            .fetch_add(1u);
    }
}

}  // namespace traccc::device
