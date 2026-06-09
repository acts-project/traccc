/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/barrier.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

namespace traccc::device {

/// @brief Create candidate edges between node pairs in compatible (eta, phi)
/// bins.
///
/// One CUDA block handles one bin-pair task.  Threads stage a chunk of bin-1
/// nodes into shared-memory caches (phi as a separate float array, and the
/// (tau_min, tau_max, r, z) float4 per node), then for every bin-2 node test
/// the cached chunk against geometric and kinematic cuts (gbts_checks),
/// atomically reserving a slot in the output via nEdgesCounter.
///
/// @param[in]  thread_id                     Thread/block identifier (one
///                                           block/task)
/// @param[in]  barrier                       Block-wide barrier
/// @param[in,out] phi                        Shared-mem cache: phi / node
/// @param[in,out] node_params                Shared-mem cache: (tau_min,
///                                           tau_max, r, z) float4 / node
/// @param[in]  d_bin_pair_views_view         Per-task bin1/bin2 ranges
/// @param[in]  d_bin_pair_dphi_view          Per-task dphi window
/// @param[in]  d_node_params_view            Per-node (tau_min, tau_max, r, z)
/// @param[in]  d_node_phi_view               Per-node phi
/// @param[in]  d_gbts_edge_making_params     Geometric / kinematic cuts
/// @param[in,out] nEdgesCounter              Global edge-slot atomic counter
/// @param[out] d_edge_nodes_view             (src, dst) per produced edge
/// @param[out] d_edge_params_view            Packed [exp(-eta), curv, phi_z,
///                                           phi_w] per produced edge
/// @param[out] d_num_outgoing_edges_view     Per-dst-node incoming-edge count
/// @param[in]  nMaxEdges                     Upper bound on edges (cap)
/// @param[in]  nPhiBins                      Number of phi bins per eta slice
///
/// Shared-memory scratch for graph_edge_making: a block-local copy of the
/// current node1 chunk (phi values and packed node params).
struct graph_edge_making_shared_payload {
    float* phi;
    float4* node_pack;
};

template <concepts::thread_id1 thread_id_t, concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void graph_edge_making(
    const thread_id_t& thread_id, const barrier_t& barrier,
    const graph_edge_making_shared_payload& shared,
    const collection_types<unsigned int>::const_view& d_bin_pair_views_view,
    const collection_types<float>::const_view& d_bin_pair_dphi_view,
    const collection_types<float4>::const_view& d_node_params_view,
    const collection_types<float>::const_view& d_node_phi_view,
    const gbts_edge_making_params& d_gbts_edge_making_params,
    unsigned int& nEdgesCounter,
    const collection_types<uint2>::view d_edge_nodes_view,
    const collection_types<gbts_edge4>::view d_edge_params_view,
    const collection_types<unsigned int>::view d_num_outgoing_edges_view,
    const unsigned int nMaxEdges, const unsigned int nPhiBins);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/graph_edge_making.ipp"
