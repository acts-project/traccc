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
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

namespace traccc::device {

/// @brief Scatter nodes into (eta, phi)-sorted slots and pack their geometry
/// tuple.
///
/// Each thread picks one node, atomically increments its (eta, phi) write
/// cursor in d_phi_cusums_view, packs the geometry / kinematic tuple into
/// d_node_params_view at that slot, and stores the original SP index in
/// d_node_index_view.
///
/// @param[in]  globalIndex                Current thread index
/// @param[in]  d_sp_params_view           Layer-ordered (x, y, z, r) per SP
/// @param[in]  d_node_eta_index_view      Eta-bin index per node
/// @param[in]  d_node_phi_index_view      Phi-bin index per node
/// @param[in,out] d_phi_cusums_view       Per-(eta, phi) write cursor
/// @param[out] d_node_params_view         5-tuple geometry params, sorted slot
/// @param[out] d_node_index_view          Sorted slot → original SP index
/// @param[in]  d_original_sp_idx_view     Layer-ordered → original SP map
/// @param[in]  d_tau_lut_view             Optional tau LUT (used iff
/// ap.useTauLUT)
/// @param[in]  ap                         Node-sorting / tau-prediction params
/// @param[in]  nPhiBins                   Number of phi bins per eta slice
///
TRACCC_HOST_DEVICE
inline void node_sorting(
    const global_index_t globalIndex,
    const collection_types<float4>::const_view& d_sp_params_view,
    const collection_types<unsigned int>::const_view& d_node_eta_index_view,
    const collection_types<unsigned int>::const_view& d_node_phi_index_view,
    const collection_types<unsigned int>::view d_phi_cusums_view,
    const collection_types<float4>::view d_node_params_view,
    const collection_types<float>::view d_node_phi_view,
    const collection_types<unsigned int>::view d_node_index_view,
    const collection_types<unsigned int>::const_view& d_original_sp_idx_view,
    const collection_types<float>::const_view& d_tau_lut_view,
    const gbts_node_sorting_params& ap, const unsigned int nPhiBins);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/node_sorting.ipp"
