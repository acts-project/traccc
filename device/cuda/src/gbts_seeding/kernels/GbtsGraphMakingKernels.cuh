/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// CUDA include(s)
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <vector_functions.h>

// Project include(s)
#include "traccc/cuda/gbts_seeding/gbts_seeding_algorithm.hpp"

namespace traccc::cuda::kernels {

struct __align__(8) half4 {
    __half x, y, z, w;
};

inline __device__ __host__ half4 make_half4(const __half x, const __half y,
                                            const __half z, const __half w) {
    half4 t;
    t.x = x;
    t.y = y;
    t.z = z;
    t.w = w;
    return t;
}

inline __device__ __half phi_wrap(__half phi) {
    const __half PI_2_h = __float2half(2 * CUDART_PI_F);
    const __half ONE_h = __float2half(1.0f);
    return phi - PI_2_h * hrint(phi * (ONE_h / PI_2_h));
}
inline __device__ float phi_wrap(float phi) {
    return phi -
           2.0f * CUDART_PI_F * rintf(phi * (1.0f / (2.0f * CUDART_PI_F)));
}

inline __device__ void gbts_checks(
    const float4 node_params_1, const float4 node_params_2, int2* d_edge_nodes,
    half4* d_edge_params, int* d_num_outgoing_edges, unsigned int* d_counters,
    const unsigned int globalIdx2, const unsigned int begin_bin1,
    const unsigned int n1Idx, const float phi1, const float phi2,
    const float deltaPhi, const float minDeltaRad, const float min_z0,
    const float max_z0, const float maxOuterRad, const float min_zU,
    const float max_zU, const float max_kappa, const float low_Kappa_d0,
    const float high_Kappa_d0, const unsigned int nMaxEdges) {

    const float tau_min1 = node_params_1.x;
    const float tau_max1 = node_params_1.y;
    const float r1 = node_params_1.z;
    const float z1 = node_params_1.w;
    const float tau_min2 = node_params_2.x;
    const float tau_max2 = node_params_2.y;
    const float r2 = node_params_2.z;
    const float z2 = node_params_2.w;
    const float dr = r2 - r1;

    if (dr < minDeltaRad) {
        return;
    }
    const float dz = z2 - z1;
    const float tau = dz / dr;
    const float ftau = fabsf(tau);

    if ((ftau < tau_min2) || (ftau > tau_max2)) {
        return;
    }
    if ((ftau < tau_min1) || (ftau > tau_max1)) {
        return;
    }
    // RZ doublet filter cuts
    const float z0 = z1 - r1 * tau;
    if ((z0 < min_z0) || (z0 > max_z0)) {
        return;
    }
    const float zouter = z0 + maxOuterRad * tau;

    if (zouter < min_zU || zouter > max_zU) {
        return;
    }

    const float dphi = phi_wrap(phi2 - phi1);

    if (fabsf(dphi) > deltaPhi) {
        return;
    }
    const float curv = dphi / dr;
    const float d0_for_max_curv = r1 * r2 * (fabsf(curv) - max_kappa);
    const float d0_max = (ftau < 4.0f) ? low_Kappa_d0 : high_Kappa_d0;
    if (d0_for_max_curv > d0_max) {
        return;
    }
    const unsigned int nEdges =
        atomicAdd(&d_counters[traccc::device::gbts_counter::nEdges], 1);
    if (nEdges < nMaxEdges) {
        const __half exp_eta = __float2half(sqrtf(1 + tau * tau) - tau);
        // edge linking order is inside->out
        atomicAdd(&d_num_outgoing_edges[begin_bin1 + n1Idx], 1);

        d_edge_nodes[nEdges] = make_int2(globalIdx2, begin_bin1 + n1Idx);

        d_edge_params[nEdges] = make_half4(exp_eta, __float2half(curv),
                                           __float2half(phi2 + curv * r2),
                                           __float2half(phi1 + curv * r1));
    }
}

__global__ static void graphEdgeMakingKernel(
    const uint4* d_bin_pair_views, const float* d_bin_pair_dphi,
    const float4* d_node_params, const float* d_node_phi,
    const gbts_edge_making_params edge_making_params, unsigned int* d_counters,
    int2* d_edge_nodes, half4* d_edge_params, int* d_num_outgoing_edges,
    const unsigned int nMaxEdges, const unsigned int nPhiBins) {

    const float minDeltaRad = edge_making_params.minDeltaRadius;
    const float min_z0 = edge_making_params.min_z0;
    const float max_z0 = edge_making_params.max_z0;
    const float maxOuterRad = edge_making_params.maxOuterRadius;
    const float min_zU = edge_making_params.cut_zMinU;
    const float max_zU = edge_making_params.cut_zMaxU;
    const float max_kappa = edge_making_params.max_Kappa;
    const float low_Kappa_d0 = edge_making_params.low_Kappa_d0;
    const float high_Kappa_d0 = edge_making_params.high_Kappa_d0;

    __shared__ float phi[traccc::device::gbts_consts::node_buffer_length];
    __shared__ float4
        node_params[traccc::device::gbts_consts::node_buffer_length];

    const uint4 views = d_bin_pair_views[blockIdx.x];
    const float deltaPhi = d_bin_pair_dphi[blockIdx.x];

    const unsigned int begin_bin1 = views.x;
    const unsigned int begin_bin2 = views.z;
    const unsigned int num_nodes1 = views.y - begin_bin1;
    const unsigned int num_nodes2 = views.w - begin_bin2;

    for (int idx = threadIdx.x; idx < num_nodes1; idx += blockDim.x) {
        // loading a chunk of nodes1 into shared mem buffers
        const unsigned int gidx = idx + begin_bin1;
        node_params[idx] = d_node_params[gidx];
        phi[idx] = d_node_phi[gidx];
    }

    __syncthreads();

    const float phi0 = phi[0];
    const float phiN = phi[num_nodes1 - 1];
    const float phi_bin_width = 2.0f * CUDART_PI_F / nPhiBins;
    const float break_threshold = deltaPhi + phi_bin_width - CUDART_PI_F;

    unsigned int last_n1 = 0;  // initial value for the sliding window

    for (int n2Idx = threadIdx.x; n2Idx < num_nodes2; n2Idx += blockDim.x) {

        const unsigned int globalIdx2 = begin_bin2 + n2Idx;
        const float phi2 = d_node_phi[globalIdx2];

        unsigned int n1Idx = last_n1;

        float min_phi1 = phi2 - deltaPhi;
        float max_phi1 = phi2 + deltaPhi;

        if (min_phi1 < -CUDART_PI_F) {
            min_phi1 += 2.0f * CUDART_PI_F;
        }
        if (max_phi1 > CUDART_PI_F) {
            max_phi1 -= 2.0f * CUDART_PI_F;
        }
        const bool boundary = max_phi1 < min_phi1;  // +/- pi wraparound

        // expand over nearest bin boundary
        max_phi1 += phi_bin_width;
        min_phi1 -= phi_bin_width;

        if (!boundary) {
            if (phi0 > max_phi1) {
                continue;
            }
            if (phiN < min_phi1) {
                // if bin1 can't be part of a wraparound
                // from a high-phi node skip it
                if (phi0 > break_threshold) {
                    break;
                }
                continue;
            }
        } else {
            if (phi0 < max_phi1) {
                // if not to large for lower wraparound don't skip it
                n1Idx = 0;
            } else if (phiN < min_phi1) {
                continue;
            }
        }

        const float4 np2 = d_node_params[globalIdx2];
        if (!boundary) {
            for (; n1Idx < num_nodes1; n1Idx++) {
                const float phi1 = phi[n1Idx];

                if (phi1 > max_phi1) {
                    break;
                }
                if (phi1 < min_phi1) {
                    continue;
                }
                last_n1 = n1Idx;

                const float4 np1 = node_params[n1Idx];
                gbts_checks(
                    np1, np2, d_edge_nodes, d_edge_params, d_num_outgoing_edges,
                    d_counters, globalIdx2, begin_bin1, n1Idx, phi1, phi2,
                    deltaPhi, minDeltaRad, min_z0, max_z0, maxOuterRad, min_zU,
                    max_zU, max_kappa, low_Kappa_d0, high_Kappa_d0, nMaxEdges);
            }
        } else {
            for (; n1Idx < num_nodes1; n1Idx++) {
                const float phi1 = phi[n1Idx];
                if (phi1 > max_phi1 && phi1 < min_phi1) {
                    continue;
                }
                last_n1 = n1Idx;

                const float4 np1 = node_params[n1Idx];
                gbts_checks(
                    np1, np2, d_edge_nodes, d_edge_params, d_num_outgoing_edges,
                    d_counters, globalIdx2, begin_bin1, n1Idx, phi1, phi2,
                    deltaPhi, minDeltaRad, min_z0, max_z0, maxOuterRad, min_zU,
                    max_zU, max_kappa, low_Kappa_d0, high_Kappa_d0, nMaxEdges);
            }
        }
    }
}

__global__ static void graphEdgeLinkingKernel(const int2* d_edge_nodes,
                                              int* d_edge_links,
                                              int* d_num_outgoing_edges,
                                              const unsigned int nEdges) {

    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (edge_idx >= nEdges) {
        return;
    }
    int sharedNode = d_edge_nodes[edge_idx].y;

    // this converts num_outgoing_edges to the start postion for each node in
    // d_edge_links
    int pos = atomicSub(&d_num_outgoing_edges[sharedNode], 1);
    // provides views of edges leaving the sharedNode for linking
    d_edge_links[pos - 1] = edge_idx;
}

__global__ static void graphEdgeMatchingKernel(
    const gbts_graph_matching_params graph_matching_params,
    const half4* d_edge_params, const int2* d_edge_nodes,
    const int* d_num_outgoing_edges, const int* d_edge_links,
    unsigned char* d_num_neighbours, int* d_neighbours, int* d_reIndexer,
    unsigned int* d_counters, const unsigned int nEdges,
    const unsigned int nMaxNei) {

    const __half cut_dphi_max =
        __float2half(graph_matching_params.cut_dphi_max);
    const __half cut_dcurv_max =
        __float2half(graph_matching_params.cut_dcurv_max);
    const __half cut_tau_ratio_max =
        __float2half(graph_matching_params.cut_tau_ratio_max);
    const __half PI_h = __float2half(CUDART_PI_F);
    const __half PI_2_h = __float2half(2 * CUDART_PI_F);
    const __half ONE_h = __float2half(1.0f);

    const int edge1_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (edge1_idx >= nEdges) {
        return;
    }

    const int sharedNode = d_edge_nodes[edge1_idx].x;

    const int link_begin = d_num_outgoing_edges[sharedNode];
    // the number of edges leaving the sharedNode
    const int nLinks = d_num_outgoing_edges[sharedNode + 1] - link_begin;
    if (nLinks == 0) {
        return;
    }
    const half4 params1 =
        d_edge_params[edge1_idx];  // [exp_eta, curv, Phi1, Phi2]

    const __half uat_2 = ONE_h / params1.x;
    const __half Phi2 = params1.z;
    const __half curv2 = params1.y;

    const int nei_pos = nMaxNei * edge1_idx;

    unsigned char num_nei = 0;

    for (int k = 0; k < nLinks; k++) {  // loop over potential neighbours

        if (num_nei >= nMaxNei) {
            break;
        }
        const int edge2_idx = d_edge_links[link_begin + k];

        const half4 params2 = d_edge_params[edge2_idx];

        const __half tau_ratio = params2.x * uat_2 - ONE_h;

        if (__habs(tau_ratio) > cut_tau_ratio_max) {  // bad match
            continue;
        }

        const __half dPhi = phi_wrap(Phi2 - params2.w);  // Phi2

        if (__habs(dPhi) > cut_dphi_max) {
            continue;
        }

        const __half dcurv = curv2 - params2.y;

        if (__habs(dcurv) > cut_dcurv_max) {
            continue;
        }

        d_neighbours[nei_pos + num_nei] = edge2_idx;
        d_reIndexer[edge2_idx] = 1;

        ++num_nei;
    }

    d_num_neighbours[edge1_idx] = num_nei;

    if (num_nei != 0) {
        d_reIndexer[edge1_idx] = 1;
        atomicAdd(&d_counters[traccc::device::gbts_counter::nConnections],
                  num_nei);
    }
}

__global__ void edgeReIndexingKernel(int* d_reIndexer, unsigned int* d_counters,
                                     const unsigned int nEdges) {

    // each thread gets an edge

    const int edge_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (edge_idx >= nEdges) {
        return;
    }
    if (d_reIndexer[edge_idx] == -1) {
        return;
    }
    d_reIndexer[edge_idx] =
        atomicAdd(&d_counters[traccc::device::gbts_counter::nConnectedEdges], 1);
}

__global__ static void graphCompressionKernel(
    const unsigned int* d_orig_node_index, const int2* d_edge_nodes,
    const unsigned char* d_num_neighbours, const int* d_neighbours,
    const int* d_reIndexer, int* d_output_graph, const unsigned int nEdges,
    const unsigned int nMaxNei) {

    const int edge_size = 2 + 1 + nMaxNei;

    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nEdges;
         idx += blockDim.x * gridDim.x) {

        if (idx >= nEdges) {
            continue;
        }
        const int newIdx = d_reIndexer[idx];
        if (newIdx == -1) {
            continue;
        }
        const int pos = edge_size * newIdx;
        const int2 edge_nodes = d_edge_nodes[idx];
        const int node1_idx = d_orig_node_index[edge_nodes.x];
        d_output_graph[pos + traccc::device::gbts_consts::node1] = node1_idx;
        const int node2_idx = d_orig_node_index[edge_nodes.y];
        d_output_graph[pos + traccc::device::gbts_consts::node2] = node2_idx;

        const unsigned char nNei = d_num_neighbours[idx];
        d_output_graph[pos + traccc::device::gbts_consts::nNei] = nNei;
        const int nei_pos = nMaxNei * idx;
        for (int k = 0; k < nNei; k++) {
            d_output_graph[pos + traccc::device::gbts_consts::nei_start + k] =
                d_reIndexer[d_neighbours[nei_pos + k]];
        }
    }
}

}  // namespace traccc::cuda::kernels
