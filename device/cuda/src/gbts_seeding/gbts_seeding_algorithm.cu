/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/barrier.hpp"
#include "../utils/cuda_error_handling.hpp"
#include "../utils/global_index.hpp"
#include "../utils/thread_id.hpp"
#include "../utils/utils.hpp"
#include "traccc/cuda/gbts_seeding/gbts_seeding_algorithm.hpp"

// Project include(s).
#include "traccc/gbts_seeding/device/add_terminus_to_path_store.hpp"
#include "traccc/gbts_seeding/device/bin_sp.hpp"
#include "traccc/gbts_seeding/device/cca_iteration.hpp"
#include "traccc/gbts_seeding/device/count_sp_by_layer.hpp"
#include "traccc/gbts_seeding/device/count_terminus_edges.hpp"
#include "traccc/gbts_seeding/device/edge_re_indexing.hpp"
#include "traccc/gbts_seeding/device/eta_phi_counting.hpp"
#include "traccc/gbts_seeding/device/eta_phi_prefix_sum.hpp"
#include "traccc/gbts_seeding/device/fill_path_store.hpp"
#include "traccc/gbts_seeding/device/fit_segments.hpp"
#include "traccc/gbts_seeding/device/gbts_seed_conversion.hpp"
#include "traccc/gbts_seeding/device/graph_compression.hpp"
#include "traccc/gbts_seeding/device/graph_edge_linking.hpp"
#include "traccc/gbts_seeding/device/graph_edge_making.hpp"
#include "traccc/gbts_seeding/device/graph_edge_matching.hpp"
#include "traccc/gbts_seeding/device/minmax_rad.hpp"
#include "traccc/gbts_seeding/device/node_sorting.hpp"
#include "traccc/gbts_seeding/device/reset_edge_bids.hpp"
#include "traccc/gbts_seeding/device/seeds_bid_for_hits.hpp"
#include "traccc/gbts_seeding/device/seeds_rebid_for_edges.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// System include(s).
#include <algorithm>

namespace traccc::cuda {

using float4 = traccc::float4;
using uint2 = traccc::uint2;
using int2 = traccc::int2;

namespace kernels {

// ---------------------------------------------------------------------------
// Stage 1 — nodes-making kernels
// ---------------------------------------------------------------------------

/// CUDA kernel for running @c traccc::device::count_sp_by_layer
__global__ void count_sp_by_layer(
    const edm::spacepoint_collection::const_view spacepoints,
    const edm::measurement_collection::const_view measurements,
    const collection_types<short>::const_view volumeToLayerMap,
    const collection_types<std::pair<unsigned int, unsigned int>>::const_view
        surfaceToLayerMap,
    const collection_types<char>::const_view layerType,
    const collection_types<float4>::view reducedSP,
    const collection_types<unsigned int>::view layerCounts,
    const collection_types<unsigned short>::view spacepointsLayer,
    const unsigned int nSp, const unsigned long int volumeMapSize,
    const unsigned long int surfaceMapSize,
    const gbts_sp_counting_params sp_counting_params) {

    for (unsigned int sp_idx = details::global_index1(); sp_idx < nSp;
         sp_idx += blockDim.x * gridDim.x) {
        device::count_sp_by_layer(sp_idx, spacepoints, measurements,
                                  volumeToLayerMap, surfaceToLayerMap,
                                  layerType, reducedSP, layerCounts,
                                  spacepointsLayer, volumeMapSize,
                                  surfaceMapSize, sp_counting_params);
    }
}

/// CUDA kernel for running @c traccc::device::bin_sp
__global__ void bin_sp(
    const collection_types<float4>::view sp_params,
    const collection_types<float4>::const_view reducedSP,
    const collection_types<unsigned int>::view layerCounts,
    const collection_types<unsigned short>::const_view spacepointsLayer,
    const collection_types<unsigned int>::view original_sp_idx,
    const collection_types<std::pair<unsigned int, unsigned int>>::const_view
        layer_info,
    const collection_types<std::pair<float, float>>::const_view layer_geo,
    const collection_types<unsigned int>::view node_eta_index,
    const collection_types<unsigned int>::view node_phi_index,
    const collection_types<unsigned int>::view eta_phi_histo,
    const unsigned int nSp, const unsigned int nPhiBins) {

    for (unsigned int sp_idx = details::global_index1(); sp_idx < nSp;
         sp_idx += blockDim.x * gridDim.x) {
        device::bin_sp(sp_idx, sp_params, reducedSP, layerCounts,
                       spacepointsLayer, original_sp_idx, layer_info, layer_geo,
                       node_eta_index, node_phi_index, eta_phi_histo, nPhiBins);
    }
}

/// CUDA kernel for running @c traccc::device::eta_phi_counting
__global__ void eta_phi_counting(
    const collection_types<unsigned int>::const_view eta_phi_histo,
    const collection_types<unsigned int>::view eta_node_counter,
    const collection_types<unsigned int>::view phi_cusums,
    const unsigned int nEtaBins, const unsigned int nPhiBins) {

    for (unsigned int eta_bin_idx = details::global_index1();
         eta_bin_idx < nEtaBins; eta_bin_idx += blockDim.x * gridDim.x) {
        device::eta_phi_counting(eta_bin_idx, eta_phi_histo, eta_node_counter,
                                 phi_cusums, nPhiBins);
    }
}

/// CUDA kernel for running @c traccc::device::eta_phi_prefix_sum
__global__ void eta_phi_prefix_sum(
    const collection_types<unsigned int>::const_view eta_node_counter,
    const collection_types<unsigned int>::view phi_cusums,
    const unsigned int nEtaBins, const unsigned int nPhiBins) {

    for (unsigned int eta_bin_idx = details::global_index1();
         eta_bin_idx < nEtaBins; eta_bin_idx += blockDim.x * gridDim.x) {
        device::eta_phi_prefix_sum(eta_bin_idx, eta_node_counter, phi_cusums,
                                   nPhiBins);
    }
}

/// CUDA kernel for running @c traccc::device::node_sorting
__global__ void node_sorting(
    const collection_types<float4>::const_view sp_params,
    const collection_types<unsigned int>::const_view node_eta_index,
    const collection_types<unsigned int>::const_view node_phi_index,
    const collection_types<unsigned int>::view phi_cusums,
    const collection_types<float4>::view node_params,
    const collection_types<float>::view node_phi,
    const collection_types<unsigned int>::view node_index,
    const collection_types<unsigned int>::const_view original_sp_idx,
    const collection_types<float>::const_view tau_lut,
    const gbts_node_sorting_params node_sorting_params,
    const unsigned int nNodes, const unsigned int nPhiBins) {

    for (unsigned int node_idx = details::global_index1(); node_idx < nNodes;
         node_idx += blockDim.x * gridDim.x) {
        device::node_sorting(node_idx, sp_params, node_eta_index,
                             node_phi_index, phi_cusums, node_params, node_phi,
                             node_index, original_sp_idx, tau_lut,
                             node_sorting_params, nPhiBins);
    }
}

/// CUDA kernel for running @c traccc::device::minmax_rad
__global__ void minmax_rad(
    const collection_types<unsigned int>::const_view eta_bin_views,
    const collection_types<float4>::const_view node_params,
    const collection_types<float>::view bin_rads, const unsigned int nEtaBins) {

    for (unsigned int eta_bin_idx = details::global_index1();
         eta_bin_idx < nEtaBins; eta_bin_idx += blockDim.x * gridDim.x) {
        device::minmax_rad(eta_bin_idx, eta_bin_views, node_params, bin_rads);
    }
}

// ---------------------------------------------------------------------------
// Stage 2 — graph-making kernels
// ---------------------------------------------------------------------------

/// CUDA kernel for running @c traccc::device::graph_edge_making
__global__ void graph_edge_making(
    const collection_types<unsigned int>::const_view bin_pair_views,
    const collection_types<float>::const_view bin_pair_dphi,
    const collection_types<float4>::const_view node_params,
    const collection_types<float>::const_view node_phi,
    const gbts_edge_making_params edge_making_params,
    unsigned int* nEdgesCounter, collection_types<uint2>::view edge_nodes,
    const collection_types<gbts_edge4>::view edge_params,
    const collection_types<unsigned int>::view num_outgoing_edges,
    const unsigned int nMaxEdges, const unsigned int nPhiBins) {

    __shared__ float phi[traccc::device::gbts_consts::node_buffer_length];
    __shared__ float4
        node_pack[traccc::device::gbts_consts::node_buffer_length];
    const traccc::cuda::barrier barrier;
    const details::thread_id1 thread_id;

    device::graph_edge_making(
        thread_id, barrier, {phi, node_pack}, bin_pair_views, bin_pair_dphi,
        node_params, node_phi, edge_making_params, *nEdgesCounter, edge_nodes,
        edge_params, num_outgoing_edges, nMaxEdges, nPhiBins);
}

/// CUDA kernel for running @c traccc::device::graph_edge_linking
__global__ void graph_edge_linking(
    const collection_types<uint2>::const_view edge_nodes,
    const collection_types<unsigned int>::view edge_links,
    const collection_types<unsigned int>::view num_outgoing_edges,
    const unsigned int nEdges) {

    for (unsigned int edge_idx = details::global_index1(); edge_idx < nEdges;
         edge_idx += blockDim.x * gridDim.x) {
        device::graph_edge_linking(edge_idx, edge_nodes, edge_links,
                                   num_outgoing_edges);
    }
}

/// CUDA kernel for running @c traccc::device::graph_edge_matching
__global__ void graph_edge_matching(
    const gbts_edge_matching_params edge_matching_params,
    const collection_types<gbts_edge4>::const_view edge_params,
    const collection_types<uint2>::const_view edge_nodes,
    const collection_types<unsigned int>::const_view num_outgoing_edges,
    const collection_types<unsigned int>::const_view edge_links,
    const collection_types<unsigned char>::view num_neighbours,
    const collection_types<unsigned int>::view neighbours,
    const collection_types<int>::view reIndexer,
    unsigned int* nConnectionsCounter, const unsigned int nEdges,
    const unsigned int nMaxNei) {

    for (unsigned int edge_idx = details::global_index1(); edge_idx < nEdges;
         edge_idx += blockDim.x * gridDim.x) {
        device::graph_edge_matching(edge_idx, edge_matching_params, edge_params,
                                    edge_nodes, num_outgoing_edges, edge_links,
                                    num_neighbours, neighbours, reIndexer,
                                    *nConnectionsCounter, nMaxNei);
    }
}

/// CUDA kernel for running @c traccc::device::edge_re_indexing
__global__ void edge_re_indexing(collection_types<int>::view reIndexer,
                                 unsigned int* nConnectedEdgesCounter,
                                 const unsigned int nEdges) {

    for (unsigned int edge_idx = details::global_index1(); edge_idx < nEdges;
         edge_idx += blockDim.x * gridDim.x) {
        device::edge_re_indexing(edge_idx, reIndexer, *nConnectedEdgesCounter);
    }
}

/// CUDA kernel for running @c traccc::device::graph_compression
__global__ void graph_compression(
    const collection_types<unsigned int>::const_view orig_node_index,
    const collection_types<uint2>::const_view edge_nodes,
    const collection_types<unsigned char>::const_view num_neighbours,
    const collection_types<unsigned int>::const_view neighbours,
    const collection_types<int>::const_view reIndexer,
    const collection_types<unsigned int>::view output_graph,
    const unsigned int nEdges, const unsigned int nMaxNei) {

    for (unsigned int edge_idx = details::global_index1(); edge_idx < nEdges;
         edge_idx += blockDim.x * gridDim.x) {
        device::graph_compression(edge_idx, orig_node_index, edge_nodes,
                                  num_neighbours, neighbours, reIndexer,
                                  output_graph, nMaxNei);
    }
}

// ---------------------------------------------------------------------------
// Stage 3 — graph-processing kernels
// ---------------------------------------------------------------------------

/// CUDA kernel for running @c traccc::device::cca_iteration
__global__ void cca_iteration(
    const collection_types<unsigned int>::const_view output_graph,
    const collection_types<unsigned char>::view levels,
    const collection_types<char>::view active_edges,
    const collection_types<short2>::view outgoing_paths,
    const unsigned char iter, const unsigned int nConnectedEdges,
    const unsigned int max_num_neighbours, const unsigned char minLevel) {

    for (unsigned int edge_idx = details::global_index1();
         edge_idx < nConnectedEdges; edge_idx += blockDim.x * gridDim.x) {
        device::cca_iteration(edge_idx, output_graph, levels, active_edges,
                              outgoing_paths, iter, nConnectedEdges,
                              max_num_neighbours, minLevel);
    }
}

/// CUDA kernel for running @c traccc::device::count_terminus_edges
__global__ void count_terminus_edges(
    const collection_types<short2>::view outgoing_paths,
    unsigned int* nPathsCounter, unsigned int* nPathStoreSizeCounter,
    const unsigned int nEdges) {

    for (unsigned int edge_idx = details::global_index1(); edge_idx < nEdges;
         edge_idx += blockDim.x * gridDim.x) {
        device::count_terminus_edges(edge_idx, outgoing_paths, *nPathsCounter,
                                     *nPathStoreSizeCounter);
    }
}

/// CUDA kernel for running @c traccc::device::add_terminus_to_path_store
__global__ void add_terminus_to_path_store(
    const collection_types<int2>::view path_store,
    const collection_types<short2>::const_view outgoing_paths,
    const unsigned int nEdges) {

    for (unsigned int edge_idx = details::global_index1(); edge_idx < nEdges;
         edge_idx += blockDim.x * gridDim.x) {
        device::add_terminus_to_path_store(edge_idx, path_store,
                                           outgoing_paths);
    }
}

/// CUDA kernel for running @c traccc::device::fill_path_store
__global__ void fill_path_store(
    const collection_types<int2>::view path_store,
    const collection_types<unsigned int>::const_view output_graph,
    const collection_types<unsigned char>::const_view levels,
    unsigned int* nPathStoreSizeCounter, const unsigned int nTerminus,
    const unsigned int nTerminusPerBlock, const unsigned int max_num_neighbours,
    const unsigned int nReachablePaths) {

    __shared__ traccc::uint2
        live_paths[traccc::device::gbts_consts::live_path_buffer];
    __shared__ int n_live_paths;
    const traccc::cuda::barrier barrier;
    const details::thread_id1 thread_id;

    device::fill_path_store(
        thread_id, barrier, {live_paths, n_live_paths}, path_store,
        output_graph, levels, *nPathStoreSizeCounter, nTerminus,
        nTerminusPerBlock, max_num_neighbours, nReachablePaths);
}

/// CUDA kernel for running @c traccc::device::fit_segments
__global__ void fit_segments(
    const collection_types<float4>::const_view reducedSP,
    const collection_types<unsigned int>::const_view output_graph,
    const collection_types<int2>::const_view path_store,
    const collection_types<int2>::view seed_proposals,
    const collection_types<unsigned long long int>::view edge_bids,
    const collection_types<char>::view seed_ambiguity,
    unsigned int* nPathStoreSize, unsigned int* nPropsCounter,
    const unsigned int nTerminusEdges, const unsigned char minLevel,
    const unsigned int max_num_neighbours,
    const gbts_seed_extraction_params seed_extraction_params,
    const float max_z0) {

    const unsigned int store_size = *nPathStoreSize;
    for (unsigned int path_idx = details::global_index1();
         path_idx + nTerminusEdges < store_size;
         path_idx += blockDim.x * gridDim.x) {
        device::fit_segments(
            path_idx, reducedSP, output_graph, path_store, seed_proposals,
            edge_bids, seed_ambiguity, *nPropsCounter, nTerminusEdges, minLevel,
            max_num_neighbours, seed_extraction_params, max_z0);
    }
}

/// CUDA kernel for running @c traccc::device::reset_edge_bids
__global__ void reset_edge_bids(
    const collection_types<int2>::const_view path_store,
    const collection_types<int2>::view seed_proposals,
    const collection_types<unsigned long long int>::view edge_bids,
    const collection_types<char>::view seed_ambiguity,
    const unsigned int nProps, unsigned int* nRejectedPropsCounter) {

    for (unsigned int prop_idx = details::global_index1(); prop_idx < nProps;
         prop_idx += blockDim.x * gridDim.x) {
        device::reset_edge_bids(prop_idx, path_store, seed_proposals, edge_bids,
                                seed_ambiguity, *nRejectedPropsCounter);
    }
}

/// CUDA kernel for running @c traccc::device::seeds_rebid_for_edges
__global__ void seeds_rebid_for_edges(
    const collection_types<int2>::const_view path_store,
    const collection_types<int2>::view seed_proposals,
    const collection_types<unsigned long long int>::view edge_bids,
    const collection_types<char>::view seed_ambiguity,
    const unsigned int nProps, unsigned int* nRejectedPropsCounter,
    const bool first_round) {

    for (unsigned int prop_idx = details::global_index1(); prop_idx < nProps;
         prop_idx += blockDim.x * gridDim.x) {
        device::seeds_rebid_for_edges(prop_idx, path_store, seed_proposals,
                                      edge_bids, seed_ambiguity,
                                      *nRejectedPropsCounter, first_round);
    }
}

/// CUDA kernel for running @c traccc::device::seeds_bid_for_hits
__global__ void seeds_bid_for_hits(
    const collection_types<unsigned int>::const_view output_graph,
    const collection_types<int2>::const_view seed_proposals,
    const collection_types<int2>::const_view path_store,
    const collection_types<char>::const_view seed_ambiguity,
    const collection_types<unsigned long long int>::view hit_bids,
    const unsigned int nProps, const unsigned int edge_size) {

    for (unsigned int prop_idx = details::global_index1(); prop_idx < nProps;
         prop_idx += blockDim.x * gridDim.x) {
        device::seeds_bid_for_hits(prop_idx, output_graph, seed_proposals,
                                   path_store, seed_ambiguity, hit_bids,
                                   edge_size);
    }
}

/// CUDA kernel for running @c traccc::device::gbts_seed_conversion
__global__ void gbts_seed_conversion(
    const collection_types<int2>::const_view seed_proposals,
    const collection_types<char>::const_view seed_ambiguity,
    const collection_types<int2>::const_view path_store,
    const collection_types<unsigned int>::const_view output_graph,
    const collection_types<float4>::const_view sp_params,
    const edm::seed_collection::view output_seeds,
    const collection_types<unsigned long long int>::view hit_bids,
    const unsigned int nProps, const unsigned int max_num_neighbours,
    const float dcurv_cut_m, const float force_dropout_max_curv_m,
    const float best_hit_frac, const float tight_bid_cot_threshold,
    const bool use_dropout) {

    for (unsigned int prop_idx = details::global_index1(); prop_idx < nProps;
         prop_idx += blockDim.x * gridDim.x) {
        device::gbts_seed_conversion(
            prop_idx, seed_proposals, seed_ambiguity, path_store, output_graph,
            sp_params, output_seeds, hit_bids, max_num_neighbours, dcurv_cut_m,
            force_dropout_max_curv_m, best_hit_frac, tight_bid_cot_threshold,
            use_dropout);
    }
}

}  // namespace kernels

// ===========================================================================
// gbts_seeding_algorithm: kernel launchers
// ===========================================================================

gbts_seeding_algorithm::gbts_seeding_algorithm(
    const gbts_seedfinder_config& cfg, const memory_resource& mr,
    vecmem::copy& copy, cuda::stream& str, std::unique_ptr<const Logger> logger)
    : device::gbts_seeding_algorithm(cfg, mr, copy, std::move(logger)),
      cuda::algorithm_base{str} {}

void gbts_seeding_algorithm::count_sp_by_layer_kernel(
    const count_sp_by_layer_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nSp - 1) / n_threads;
    kernels::count_sp_by_layer<<<n_blocks, n_threads, 0,
                                 details::get_stream(stream())>>>(
        payload.spacepoints, payload.measurements, payload.volumeToLayerMap,
        payload.surfaceToLayerMap, payload.layerType, payload.reducedSP,
        payload.layerCounts, payload.spacepointsLayer, payload.nSp,
        payload.volumeMapSize, payload.surfaceMapSize,
        payload.sp_counting_params);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::bin_sp_kernel(
    const bin_sp_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nSp - 1) / n_threads;
    kernels::bin_sp<<<n_blocks, n_threads, 0, details::get_stream(stream())>>>(
        payload.sp_params, payload.reducedSP, payload.layerCounts,
        payload.spacepointsLayer, payload.original_sp_idx, payload.layer_info,
        payload.layer_geo, payload.node_eta_index, payload.node_phi_index,
        payload.eta_phi_histo, payload.nSp, payload.nPhiBins);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::eta_phi_counting_kernel(
    const eta_phi_counting_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nEtaBins - 1) / n_threads;
    kernels::eta_phi_counting<<<n_blocks, n_threads, 0,
                                details::get_stream(stream())>>>(
        payload.eta_phi_histo, payload.eta_node_counter, payload.phi_cusums,
        payload.nEtaBins, payload.nPhiBins);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::eta_phi_prefix_sum_kernel(
    const eta_phi_prefix_sum_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nEtaBins - 1) / n_threads;
    kernels::eta_phi_prefix_sum<<<n_blocks, n_threads, 0,
                                  details::get_stream(stream())>>>(
        payload.eta_node_counter, payload.phi_cusums, payload.nEtaBins,
        payload.nPhiBins);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::node_sorting_kernel(
    const node_sorting_kernel_payload& payload) const {

    const unsigned int n_threads = 256;
    const unsigned int n_blocks = 1 + (payload.nNodes - 1) / n_threads;
    kernels::
        node_sorting<<<n_blocks, n_threads, 0, details::get_stream(stream())>>>(
            payload.sp_params, payload.node_eta_index, payload.node_phi_index,
            payload.phi_cusums, payload.node_params, payload.node_phi,
            payload.node_index, payload.original_sp_idx, payload.tau_lut,
            payload.node_sorting_params, payload.nNodes, payload.nPhiBins);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::minmax_rad_kernel(
    const minmax_rad_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nEtaBins - 1) / n_threads;
    kernels::
        minmax_rad<<<n_blocks, n_threads, 0, details::get_stream(stream())>>>(
            payload.eta_bin_views, payload.node_params, payload.bin_rads,
            payload.nEtaBins);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::graph_edge_making_kernel(
    const graph_edge_making_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = payload.nUsedBinPairs;
    kernels::graph_edge_making<<<n_blocks, n_threads, 0,
                                 details::get_stream(stream())>>>(
        payload.bin_pair_views, payload.bin_pair_dphi, payload.node_params,
        payload.node_phi, payload.edge_making_params, payload.nEdgesCounter,
        payload.edge_nodes, payload.edge_params, payload.num_outgoing_edges,
        payload.nMaxEdges, payload.nPhiBins);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::graph_edge_linking_kernel(
    const graph_edge_linking_kernel_payload& payload) const {

    const unsigned int n_threads = 256;
    const unsigned int n_blocks = 1 + (payload.nEdges - 1) / n_threads;
    kernels::graph_edge_linking<<<n_blocks, n_threads, 0,
                                  details::get_stream(stream())>>>(
        payload.edge_nodes, payload.edge_links, payload.num_outgoing_edges,
        payload.nEdges);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::graph_edge_matching_kernel(
    const graph_edge_matching_kernel_payload& payload) const {

    const unsigned int n_threads = 256;
    const unsigned int n_blocks = 1 + (payload.nEdges - 1) / n_threads;
    kernels::graph_edge_matching<<<n_blocks, n_threads, 0,
                                   details::get_stream(stream())>>>(
        payload.edge_matching_params, payload.edge_params, payload.edge_nodes,
        payload.num_outgoing_edges, payload.edge_links, payload.num_neighbours,
        payload.neighbours, payload.reIndexer, payload.nConnectionsCounter,
        payload.nEdges, payload.nMaxNei);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::edge_re_indexing_kernel(
    const edge_re_indexing_kernel_payload& payload) const {

    const unsigned int n_threads = 256;
    const unsigned int n_blocks = 1 + (payload.nEdges - 1) / n_threads;
    kernels::edge_re_indexing<<<n_blocks, n_threads, 0,
                                details::get_stream(stream())>>>(
        payload.reIndexer, payload.nConnectedEdgesCounter, payload.nEdges);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::graph_compression_kernel(
    const graph_compression_kernel_payload& payload) const {

    const unsigned int n_threads = 256;
    const unsigned int n_blocks = 1 + (payload.nEdges - 1) / n_threads;
    kernels::graph_compression<<<n_blocks, n_threads, 0,
                                 details::get_stream(stream())>>>(
        payload.orig_node_index, payload.edge_nodes, payload.num_neighbours,
        payload.neighbours, payload.reIndexer, payload.output_graph,
        payload.nEdges, payload.nMaxNei);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::cca_iteration_kernel(
    const cca_iteration_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nConnectedEdges - 1) / n_threads;

    kernels::cca_iteration<<<n_blocks, n_threads, 0,
                             details::get_stream(stream())>>>(
        payload.output_graph, payload.levels, payload.active_edges,
        payload.outgoing_paths, payload.iter, payload.nConnectedEdges,
        payload.max_num_neighbours, payload.minLevel);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::count_terminus_edges_kernel(
    const count_terminus_edges_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nConnectedEdges - 1) / n_threads;
    kernels::count_terminus_edges<<<n_blocks, n_threads, 0,
                                    details::get_stream(stream())>>>(
        payload.outgoing_paths, payload.nPathsCounter,
        payload.nPathStoreSizeCounter, payload.nConnectedEdges);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::add_terminus_to_path_store_kernel(
    const add_terminus_to_path_store_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nConnectedEdges - 1) / n_threads;
    kernels::add_terminus_to_path_store<<<n_blocks, n_threads, 0,
                                          details::get_stream(stream())>>>(
        payload.path_store, payload.outgoing_paths, payload.nConnectedEdges);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::fill_path_store_kernel(
    const fill_path_store_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int pathsPerTerminus =
        1 + (payload.nPaths - 1) / payload.nTerminusEdges;
    const unsigned int terminusPerBlock = std::min(
        n_threads, 1 + (traccc::device::gbts_consts::live_path_buffer - 1) /
                           pathsPerTerminus);
    const unsigned int n_blocks =
        1 + (payload.nTerminusEdges - 1) / terminusPerBlock;
    kernels::fill_path_store<<<n_blocks, n_threads, 0,
                               details::get_stream(stream())>>>(
        payload.path_store, payload.output_graph, payload.levels,
        payload.nPathStoreSizeCounter, payload.nTerminusEdges, terminusPerBlock,
        payload.max_num_neighbours, payload.nPaths + payload.nTerminusEdges);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::fit_segments_kernel(
    const fit_segments_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nPaths - 1) / n_threads;
    kernels::
        fit_segments<<<n_blocks, n_threads, 0, details::get_stream(stream())>>>(
            payload.reducedSP, payload.output_graph, payload.path_store,
            payload.seed_proposals, payload.edge_bids, payload.seed_ambiguity,
            payload.nPathStoreSize, payload.nPropsCounter,
            payload.nTerminusEdges, payload.minLevel,
            payload.max_num_neighbours, payload.seed_extraction_params,
            payload.max_z0);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::reset_edge_bids_kernel(
    const reset_edge_bids_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nProps - 1) / n_threads;
    kernels::reset_edge_bids<<<n_blocks, n_threads, 0,
                               details::get_stream(stream())>>>(
        payload.path_store, payload.seed_proposals, payload.edge_bids,
        payload.seed_ambiguity, payload.nProps, payload.nRejectedPropsCounter);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::seeds_rebid_for_edges_kernel(
    const seeds_rebid_for_edges_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nProps - 1) / n_threads;
    kernels::seeds_rebid_for_edges<<<n_blocks, n_threads, 0,
                                     details::get_stream(stream())>>>(
        payload.path_store, payload.seed_proposals, payload.edge_bids,
        payload.seed_ambiguity, payload.nProps, payload.nRejectedPropsCounter,
        payload.first_round);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::seeds_bid_for_hits_kernel(
    const seeds_bid_for_hits_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nProps - 1) / n_threads;
    kernels::seeds_bid_for_hits<<<n_blocks, n_threads, 0,
                                  details::get_stream(stream())>>>(
        payload.output_graph, payload.seed_proposals, payload.path_store,
        payload.seed_ambiguity, payload.hit_bids, payload.nProps,
        payload.edge_size);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_seed_conversion_kernel(
    const gbts_seed_conversion_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nProps - 1) / n_threads;
    kernels::gbts_seed_conversion<<<n_blocks, n_threads, 0,
                                    details::get_stream(stream())>>>(
        payload.seed_proposals, payload.seed_ambiguity, payload.path_store,
        payload.output_graph, payload.reducedSP, payload.output_seeds,
        payload.hit_bids, payload.nProps, payload.max_num_neighbours,
        payload.seed_ambi_params.dropout_dcurv_m,
        payload.seed_ambi_params.force_dropout_max_curv_m,
        payload.seed_ambi_params.best_hit_frac,
        payload.seed_ambi_params.tight_bid_cot_threshold,
        payload.seed_ambi_params.use_dropout);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

}  // namespace traccc::cuda
