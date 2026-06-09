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

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void eta_phi_counting(
    const global_index_t globalIndex,
    const collection_types<unsigned int>::const_view& d_histo_view,
    const collection_types<unsigned int>::view& d_eta_node_counter_view,
    const collection_types<unsigned int>::view& d_phi_cusums_view,
    const unsigned int nPhiBins) {

    const collection_types<unsigned int>::const_device d_histo(d_histo_view);
    collection_types<unsigned int>::device d_eta_node_counter(
        d_eta_node_counter_view);
    collection_types<unsigned int>::device d_phi_cusums(d_phi_cusums_view);

    const unsigned int offset = nPhiBins * globalIndex;

    unsigned int sum = 0;
    for (unsigned int phiIdx = 0; phiIdx < nPhiBins; phiIdx++) {
        d_phi_cusums[offset + phiIdx] = sum;
        sum += d_histo[offset + phiIdx];
    }
    d_eta_node_counter[globalIndex] = sum;
}

}  // namespace traccc::device
