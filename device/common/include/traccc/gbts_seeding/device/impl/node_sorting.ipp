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

// System include(s).
#include <cmath>
#include <cstdint>

namespace traccc::device {

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
    const gbts_node_sorting_params& ap, const unsigned int nPhiBins) {

    const collection_types<float4>::const_device d_sp_params(d_sp_params_view);
    const collection_types<unsigned int>::const_device d_node_eta_index(
        d_node_eta_index_view);
    const collection_types<unsigned int>::const_device d_node_phi_index(
        d_node_phi_index_view);
    collection_types<unsigned int>::device d_phi_cusums(d_phi_cusums_view);
    collection_types<float4>::device d_node_params(d_node_params_view);
    collection_types<float>::device d_node_phi(d_node_phi_view);
    collection_types<unsigned int>::device d_node_index(d_node_index_view);
    const collection_types<unsigned int>::const_device d_original_sp_idx(
        d_original_sp_idx_view);
    const collection_types<float>::const_device d_tau_lut(d_tau_lut_view);

    const float4 sp = d_sp_params[globalIndex];

    const float Phi = atan2f(sp.y, sp.x);
    const float r = sqrtf(sp.x * sp.x + sp.y * sp.y);
    const float z = sp.z;

    // Default to the full |tau| acceptance for nodes that carry no usable
    // cluster width (sp.w <= 0); the per-edge cuts then rely on these bounds.
    float min_tau = 0.0f;
    float max_tau = ap.maxTau;

    if (sp.w > 0) {  // type 0 only
        if (ap.useTauLUT) {
            // LUT is laid out as [w_bin_edge, min_tau_0, max_tau_0, min_tau_1,
            // max_tau_1] per bin.
            const int tau_bin =
                5 * static_cast<int>(floorf(ap.tau_lut_inv_bin * sp.w) - 1.0f);
            if (tau_bin > -1 && tau_bin < static_cast<int>(ap.tauLutSize)) {
                min_tau = d_tau_lut[static_cast<unsigned int>(tau_bin) + 1u];
                max_tau = d_tau_lut[static_cast<unsigned int>(tau_bin) + 2u];
            }
            if (max_tau < 0.0f) {
                max_tau = ap.maxTau;
            }
            if (min_tau < 0.0f) {
                min_tau = 0.0f;
            }
        } else {
            // linear fit + correction for short clusters
            min_tau = ap.tMin_slope * (sp.w - ap.offset);
            max_tau = ap.tMax_min + ap.tMax_correction / (sp.w + ap.offset) +
                      ap.tMax_slope * (sp.w - ap.offset);
        }
    }

    const unsigned int eta_index = d_node_eta_index[globalIndex];
    const unsigned int histo_bin =
        d_node_phi_index[globalIndex] + nPhiBins * eta_index;

    const unsigned int pos =
        vecmem::device_atomic_ref<unsigned int>(d_phi_cusums[histo_bin])
            .fetch_add(1);

    d_node_params[pos] = make_float4(min_tau, max_tau, r, z);
    d_node_phi[pos] = Phi;
    d_node_index[pos] = d_original_sp_idx[globalIndex];
}

}  // namespace traccc::device
