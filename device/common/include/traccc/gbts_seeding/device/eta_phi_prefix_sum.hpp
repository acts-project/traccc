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

namespace traccc::device {

/// @brief Convert the per-(eta, phi) sums into cumulative offsets.
///
/// One thread per eta-bin walks its phi row and rewrites it as a running
/// cumulative sum starting from the per-eta global offset, yielding a flat
/// write-cursor table for node_sorting.
///
/// @param[in]  globalIndex                Current thread index
/// @param[in]  d_eta_node_counter_view    Per-eta global offset (prefix sum)
/// @param[in,out] d_phi_cusums_view       Per-(eta, phi) sums in, offsets out
/// @param[in]  nPhiBins                   Number of phi bins per eta slice
///
TRACCC_HOST_DEVICE
inline void eta_phi_prefix_sum(
    const global_index_t globalIndex,
    const collection_types<int>::const_view& d_eta_node_counter_view,
    const collection_types<int>::view d_phi_cusums_view,
    const unsigned int nPhiBins);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/eta_phi_prefix_sum.ipp"
