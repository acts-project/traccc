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

/// @brief Sum the (eta, phi) histogram across phi for each eta bin.
///
/// One thread per eta-bin walks its phi row, writes the running per-phi sum
/// into d_phi_cusums_view (later turned cumulative), and stores the total
/// for that eta in d_eta_node_counter_view.
///
/// @param[in]  globalIndex                Current thread index
/// @param[in]  d_histo_view               Flat (eta, phi) node histogram
/// @param[out] d_eta_node_counter_view    Per-eta total node count
/// @param[out] d_phi_cusums_view          Per-(eta, phi) prefix-sum scratch
/// @param[in]  nPhiBins                   Number of phi bins per eta slice
///
TRACCC_HOST_DEVICE
inline void eta_phi_counting(
    const global_index_t globalIndex,
    const collection_types<int>::const_view& d_histo_view,
    const collection_types<int>::view d_eta_node_counter_view,
    const collection_types<int>::view d_phi_cusums_view,
    const unsigned int nPhiBins);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/eta_phi_counting.ipp"
