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

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Global Event Data) Payload for the @c
/// traccc::device::gbts_prefix_sum_eta_phi_bins function
struct gbts_prefix_sum_eta_phi_bins_payload {
    /// Number of eta bins
    unsigned int nEtaBins;
    /// Number of phi bins per eta slice
    unsigned int nPhiBins;
    /// Per-eta prefix-summed offsets into the global node array
    vecmem::data::vector_view<const unsigned int> eta_node_counter;
    /// In/out: per-eta phi prefix sums, made cumulative within each eta
    vecmem::data::vector_view<unsigned int> phi_cusums;
};

/// @brief Convert the per-(eta, phi) sums into cumulative offsets.
///
/// One thread per eta-bin walks its phi row and rewrites it as a running
/// cumulative sum starting from the per-eta global offset, yielding a flat
/// write-cursor table for gbts_sort_nodes.
///
/// @param[in] globalIndex Current thread index
/// @param[in] payload     The global memory payload
///
TRACCC_HOST_DEVICE
inline void gbts_prefix_sum_eta_phi_bins(
    const global_index_t globalIndex,
    const gbts_prefix_sum_eta_phi_bins_payload& payload);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/gbts_prefix_sum_eta_phi_bins.ipp"
