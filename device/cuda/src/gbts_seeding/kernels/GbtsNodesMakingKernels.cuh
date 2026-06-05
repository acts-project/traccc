/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// CUDA include(s)
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <vector_functions.h>

// Project include(s)
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"

// Detray include(s).
#include <detray/geometry/identifier.hpp>

namespace traccc::cuda::kernels {

__global__ void count_sp_by_layer_kernel(
    const traccc::edm::spacepoint_collection::const_view spacepoints_view,
    const edm::measurement_collection::const_view measurements_view,
    const short* volumeToLayerMap, const uint2* surfaceToLayerMap,
    const char* d_layerType, float4* reducedSP, unsigned int* d_layerCounts,
    short* spacepointsLayer, const unsigned int nSp,
    const unsigned long int volumeMapSize,
    const unsigned long int surfaceMapSize, const gbts_sp_counting_params ap) {

    const edm::measurement_collection::const_device measurements(
        measurements_view);
    const traccc::edm::spacepoint_collection::const_device spacepoints(
        spacepoints_view);

    for (int spIdx = threadIdx.x + blockDim.x * blockIdx.x; spIdx < nSp;
         spIdx += blockDim.x * gridDim.x) {
        // get the layer of the spacepoint
        const traccc::edm::spacepoint_collection::const_device::const_proxy_type
            spacepoint = spacepoints.at(spIdx);
        const auto measurement =
            measurements.at(spacepoint.measurement_index_1());

        detray::geometry::identifier geo_id = measurement.surface_link();
        const unsigned int volume_id = geo_id.volume();
        const short begin_or_bin = (volume_id < volumeMapSize)
                                       ? volumeToLayerMap[volume_id]
                                       : SHRT_MAX;
        // some volume_ids map one to one with layer others need searching
        if (begin_or_bin == SHRT_MAX) {
            reducedSP[spIdx].w = -CHAR_MAX - 1;
            continue;  // unconfigured volume
        }
        unsigned int layerIdx;
        if (begin_or_bin < 0) {
            unsigned int surface_index =
                static_cast<unsigned int>(geo_id.index());

            for (unsigned int surface = -1 * (begin_or_bin + 1);
                 surface < surfaceMapSize; surface++) {

                const uint2 surfaceBinPair = surfaceToLayerMap[surface];
                if (surfaceBinPair.x == surface_index) {
                    layerIdx = surfaceBinPair.y;
                    break;
                }
            }
        } else {
            layerIdx = static_cast<unsigned int>(begin_or_bin);
        }
        float cluster_diameter = measurement.diameter();
        const int type = static_cast<int>(d_layerType[layerIdx]);
        if (type == 1 && cluster_diameter > ap.type1_max_width) {
            //-ve cluster_diameter to skip cot(theta) prediction
            // large -ve to skip spacepoint entirely
            reducedSP[spIdx].w = -CHAR_MAX - 1;
            continue;
        }
        cluster_diameter = (ap.doTauCut && type != 0)
                               ? static_cast<float>(-1 * type)
                               : cluster_diameter;

        // count and store x,y,z,cw info
        atomicAdd(&d_layerCounts[layerIdx], 1);
        spacepointsLayer[spIdx] = layerIdx;
        const std::array<float, 3u> pos = spacepoint.global();
        reducedSP[spIdx] =
            make_float4(pos[0], pos[1], pos[2], cluster_diameter);
    }
}

__global__ void bin_sp_kernel(
    float4* d_sp_params, const float4* d_reducedSP, unsigned int* d_layerCounts,
    short* d_spacepointsLayer, unsigned int* d_original_sp_idx,
    const int2* d_layer_info, const float2* d_layer_geo,
    unsigned int* d_node_eta_index, unsigned int* d_node_phi_index,
    unsigned int* d_eta_phi_histo, const unsigned int nSp,
    const unsigned int nPhiBins) {

    for (int spIdx = threadIdx.x + blockDim.x * blockIdx.x; spIdx < nSp;
         spIdx += blockDim.x * gridDim.x) {

        const float4 sp = d_reducedSP[spIdx];
        if (sp.w < -CHAR_MAX) {
            continue;
        }

        const short layerIdx = d_spacepointsLayer[spIdx];
        const unsigned int binedIdx =
            atomicSub(&d_layerCounts[layerIdx], 1) - 1;
        d_original_sp_idx[binedIdx] = spIdx;
        d_sp_params[binedIdx] = sp;

        const int2 layerInfo = d_layer_info[layerIdx];
        const int bin0 = layerInfo.x;
        const int num_eta_bins = layerInfo.y;
        unsigned int etaIdx;
        if (num_eta_bins == 1) {
            etaIdx = bin0;
        } else {
            const float2 layerGeo = d_layer_geo[layerIdx];
            const float min_eta = layerGeo.x;
            const float eta_bin_width = layerGeo.y;
            const float r = sqrtf(sp.x * sp.x + sp.y * sp.y);
            const float t1 = sp.z / r;
            const float eta = -logf(sqrtf(1 + t1 * t1) - t1);
            const unsigned int binIdx = static_cast<unsigned int>(fmaxf(
                0.0f,
                fminf((eta - min_eta) / eta_bin_width, num_eta_bins - 1.0f)));
            etaIdx = bin0 + binIdx;
        }
        d_node_eta_index[binedIdx] = etaIdx;

        const float invPhiSliceWidth =
            1.0f / (2.0f * CUDART_PI_F / static_cast<float>(nPhiBins));
        const float phi = atan2f(sp.y, sp.x);
        unsigned int phiIdx =
            static_cast<unsigned int>((phi + CUDART_PI_F) * invPhiSliceWidth);

        if (phiIdx >= nPhiBins) {
            phiIdx -= nPhiBins;
        }
        d_node_phi_index[binedIdx] = phiIdx;

        const unsigned int histoBin = etaIdx * nPhiBins + phiIdx;
        atomicAdd(&d_eta_phi_histo[histoBin], 1);
    }
}

__global__ void eta_phi_counting_kernel(unsigned int* d_eta_phi_histo,
                                        unsigned int* d_eta_node_counter,
                                        unsigned int* d_phi_cusums,
                                        const unsigned int nEtaBins,
                                        const unsigned int nPhiBins) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < nEtaBins;
         idx += blockDim.x * gridDim.x) {

        const unsigned int offset = nPhiBins * idx;
        unsigned int sum = 0;
        for (unsigned int phiIdx = 0; phiIdx < nPhiBins; phiIdx++) {
            d_phi_cusums[offset + phiIdx] = sum;
            sum += d_eta_phi_histo[offset + phiIdx];
        }
        d_eta_node_counter[idx] = sum;
    }
}

__global__ void eta_phi_prefix_sum_kernel(
    const unsigned int* d_eta_node_counter, unsigned int* d_phi_cusums,
    const unsigned int nEtaBins, const unsigned int nPhiBins) {

    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < nEtaBins;
         idx += blockDim.x * gridDim.x) {

        if (idx == 0) {
            continue;
        }
        int offset = nPhiBins * idx;

        int val0 = d_eta_node_counter[idx - 1];

        for (int phiIdx = 0; phiIdx < nPhiBins; phiIdx++) {
            d_phi_cusums[offset + phiIdx] += val0;
        }
    }
}

__global__ void node_sorting_kernel(
    const float4* d_sp_params, const unsigned int* d_node_eta_index,
    const unsigned int* d_node_phi_index, unsigned int* d_phi_cusums,
    float4* d_node_params, float* d_node_phi, unsigned int* d_node_index,
    unsigned int* d_original_sp_idx, float* d_tau_lut,
    const gbts_node_sorting_params ap, const unsigned int nNodes,
    const unsigned int nPhiBins) {

    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nNodes;
         idx += blockDim.x * gridDim.x) {

        float4 sp = d_sp_params[idx];

        float Phi = atan2f(sp.y, sp.x);
        float r = sqrtf(sp.x * sp.x + sp.y * sp.y);
        float z = sp.z;

        float min_tau = 0.0f;
        float max_tau = ap.maxTau;

        if (sp.w > 0) {  // type 0 only
            if (ap.useTauLUT) {
                const int tau_bin =
                    5 * static_cast<int>(floorf(ap.tau_lut_inv_bin * sp.w) - 1);
                if (tau_bin > -1 && tau_bin < ap.tauLutSize) {
                    min_tau = d_tau_lut[tau_bin + 1];
                    max_tau =
                        d_tau_lut[tau_bin +
                                  2];  // This assumes the LUT is structured as
                                       // [w_bin_edge, min_tau, max_tau, ...]
                                       // for each bin
                }
                if (max_tau < 0.0f) {
                    max_tau = ap.maxTau;  // This I want to remove by changing
                                          // the LUT construction
                }
                if (min_tau < 0.0f) {
                    min_tau = 0.0f;  // This I want to remove by changing the
                                     // LUT construction
                }
            } else {
                // linear fit
                min_tau = ap.tMinSlope * (sp.w - ap.offset);
                // linear fit + correction for short clusters
                max_tau = ap.tMaxMin + ap.tMaxCorrection / (sp.w + ap.offset) +
                          ap.tMaxSlope * (sp.w - ap.offset);
            }
        }

        int eta_index = d_node_eta_index[idx];
        int histo_bin = d_node_phi_index[idx] + nPhiBins * eta_index;

        int pos = atomicAdd(&d_phi_cusums[histo_bin], 1);

        d_node_params[pos] = make_float4(min_tau, max_tau, r, z);
        d_node_phi[pos] = Phi;
        // keep the original index of the input spacepoint
        d_node_index[pos] = d_original_sp_idx[idx];
    }
}

__global__ void minmax_rad_kernel(const int* d_eta_bin_views,
                                  const float4* d_node_params,
                                  float* d_bins_rads,
                                  const unsigned int nEtaBins) {

    for (int globalIdx = threadIdx.x + blockDim.x * blockIdx.x;
         globalIdx < nEtaBins; globalIdx += blockDim.x * gridDim.x) {
        int node_start = d_eta_bin_views[2 * globalIdx];
        int node_end = d_eta_bin_views[2 * globalIdx + 1];

        float minR = 1e8;
        float maxR = -1e8;

        for (int idx = node_start; idx < node_end; idx++) {
            float r = d_node_params[idx].z;
            maxR = fmaxf(maxR, r);
            minR = fminf(minR, r);
        }

        d_bins_rads[2 * globalIdx] = minR;
        d_bins_rads[2 * globalIdx + 1] = maxR;
    }
}

}  // namespace traccc::cuda::kernels
