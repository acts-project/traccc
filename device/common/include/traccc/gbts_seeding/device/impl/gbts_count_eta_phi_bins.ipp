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

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void gbts_count_eta_phi_bins(
    const global_index_t globalIndex,
    const gbts_count_eta_phi_bins_payload& payload) {

    const vecmem::device_vector<const unsigned int> d_histo(
        payload.eta_phi_histo);
    vecmem::device_vector<unsigned int> d_eta_node_counter(
        payload.eta_node_counter);
    vecmem::device_vector<unsigned int> d_phi_cusums(payload.phi_cusums);

    const unsigned int offset = payload.nPhiBins * globalIndex;

    unsigned int sum = 0;
    for (unsigned int phiIdx = 0; phiIdx < payload.nPhiBins; phiIdx++) {
        d_phi_cusums[offset + phiIdx] = sum;
        sum += d_histo[offset + phiIdx];
    }
    d_eta_node_counter[globalIndex] = sum;
}

}  // namespace traccc::device
