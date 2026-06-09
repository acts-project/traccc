/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
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

// System include(s).
#include <climits>
#include <cmath>
#include <cstdint>
#include <utility>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void bin_sp(
    const global_index_t globalIndex,
    const collection_types<float4>::view sp_params_view,
    const collection_types<float4>::const_view& reducedSP_view,
    const collection_types<unsigned int>::view layerCounts_view,
    const collection_types<unsigned short>::const_view& spacepointsLayer_view,
    const collection_types<unsigned int>::view original_sp_idx_view,
    const collection_types<std::pair<unsigned int, unsigned int>>::const_view&
        layer_info_view,
    const collection_types<std::pair<float, float>>::const_view& layer_geo_view,
    const collection_types<unsigned int>::view node_eta_index_view,
    const collection_types<unsigned int>::view node_phi_index_view,
    const collection_types<unsigned int>::view eta_phi_histo_view,
    const unsigned int nPhiBins) {

    const collection_types<float4>::const_device reducedSP(reducedSP_view);
    const collection_types<unsigned short>::const_device spacepointsLayer(
        spacepointsLayer_view);
    collection_types<unsigned int>::device layerCounts(layerCounts_view);
    collection_types<float4>::device sp_params(sp_params_view);
    collection_types<unsigned int>::device original_sp_idx(
        original_sp_idx_view);
    const collection_types<std::pair<unsigned int, unsigned int>>::const_device
        d_layer_info(layer_info_view);
    const collection_types<std::pair<float, float>>::const_device d_layer_geo(
        layer_geo_view);
    collection_types<unsigned int>::device d_node_eta_index(
        node_eta_index_view);
    collection_types<unsigned int>::device d_node_phi_index(
        node_phi_index_view);
    collection_types<unsigned int>::device d_eta_phi_histo(eta_phi_histo_view);

    // --- Stage 1: bin_sp_by_layer ------------------------------------------
    const float4 sp = reducedSP[globalIndex];
    if (sp.w < -CHAR_MAX) {
        return;
    }
    const unsigned short layerIdx = spacepointsLayer[globalIndex];
    const unsigned int binedIdx =
        vecmem::device_atomic_ref<unsigned int>(layerCounts[layerIdx])
            .fetch_sub(1) -
        1u;
    original_sp_idx[binedIdx] = globalIndex;
    sp_params[binedIdx] = sp;

    // --- Stage 2: node_eta_binning -----------
    const unsigned int layerIdx_u = static_cast<unsigned int>(layerIdx);
    const std::pair<unsigned int, unsigned int> layerInfo =
        d_layer_info[layerIdx_u];
    const unsigned int bin0 = layerInfo.first;
    const unsigned int num_eta_bins = layerInfo.second;
    unsigned int eta_index;
    if (num_eta_bins == 1u) {
        eta_index = bin0;
    } else {
        const std::pair<float, float> layerGeo = d_layer_geo[layerIdx_u];
        const float min_eta = layerGeo.first;
        const float eta_bin_width = layerGeo.second;
        const float r = sqrtf(sp.x * sp.x + sp.y * sp.y);
        const float t1 = sp.z / r;
        const float eta = -logf(sqrtf(1.0f + t1 * t1) - t1);
        const unsigned int binIdx = static_cast<unsigned int>(
            fmaxf(0.0f, fminf((eta - min_eta) / eta_bin_width,
                              static_cast<float>(num_eta_bins - 1u))));
        eta_index = bin0 + binIdx;
    }
    d_node_eta_index[binedIdx] = eta_index;

    // --- Stage 3: eta_phi_histo --------------
    const float inv_phiSliceWidth =
        1.0f / (traccc::device::TWO_PI_F / static_cast<float>(nPhiBins));
    const float Phi = atan2f(sp.y, sp.x);
    unsigned int phiIdx = static_cast<unsigned int>(
        (Phi + traccc::device::PI_F) * inv_phiSliceWidth);
    if (phiIdx >= nPhiBins) {
        phiIdx -= nPhiBins;
    }
    d_node_phi_index[binedIdx] = phiIdx;

    const unsigned int histo_bin = phiIdx + nPhiBins * eta_index;
    vecmem::device_atomic_ref<unsigned int>(d_eta_phi_histo[histo_bin])
        .fetch_add(1u);
}

}  // namespace traccc::device
