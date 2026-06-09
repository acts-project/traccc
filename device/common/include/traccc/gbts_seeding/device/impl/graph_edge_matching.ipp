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

namespace traccc::device {

// For each edge, find up to nMaxNei compatible neighbour edges sharing its
// outer node, recording them and marking both edges as "kept". The edge
// parameters are read from the packed gbts_edge4 buffer ([exp_eta, curv, phi_z,
// phi_w]) and the cuts are evaluated in float.
TRACCC_HOST_DEVICE inline void graph_edge_matching(
    const global_index_t globalIndex,
    const gbts_edge_matching_params& d_graph_matching_params,
    const collection_types<gbts_edge4>::const_view& d_edge_params_view,
    const collection_types<uint2>::const_view& d_edge_nodes_view,
    const collection_types<unsigned int>::const_view& d_num_outgoing_edges_view,
    const collection_types<unsigned int>::const_view& d_edge_links_view,
    const collection_types<unsigned char>::view& d_num_neighbours_view,
    const collection_types<unsigned int>::view& d_neighbours_view,
    const collection_types<int>::view& d_reIndexer_view,
    unsigned int& nConnectionsCounter, const unsigned int nMaxNei) {

    const collection_types<gbts_edge4>::const_device d_edge_params(
        d_edge_params_view);
    const collection_types<uint2>::const_device d_edge_nodes(d_edge_nodes_view);
    const collection_types<unsigned int>::const_device d_num_outgoing_edges(
        d_num_outgoing_edges_view);
    const collection_types<unsigned int>::const_device d_edge_links(
        d_edge_links_view);
    collection_types<unsigned char>::device d_num_neighbours(
        d_num_neighbours_view);
    collection_types<unsigned int>::device d_neighbours(d_neighbours_view);
    collection_types<int>::device d_reIndexer(d_reIndexer_view);

    const float cut_dphi_max = d_graph_matching_params.cut_dphi_max;
    const float cut_dcurv_max = d_graph_matching_params.cut_dcurv_max;
    const float cut_tau_ratio_max = d_graph_matching_params.cut_tau_ratio_max;

    const unsigned int sharedNode = d_edge_nodes[globalIndex].x;

    const unsigned int link_begin = d_num_outgoing_edges[sharedNode];
    // the number of edges leaving the sharedNode
    const unsigned int nLinks =
        d_num_outgoing_edges[sharedNode + 1u] - link_begin;
    if (nLinks == 0u) {
        return;
    }

    const gbts_edge4 params1 = d_edge_params[globalIndex];  // [exp_eta, curv,
                                                            //  phi_z, phi_w]
    const float uat_2 = 1.0f / gbts_edge_to_float(params1.x);
    const float Phi2 = gbts_edge_to_float(params1.z);
    const float curv2 = gbts_edge_to_float(params1.y);

    const unsigned int nei_pos = nMaxNei * globalIndex;

    unsigned char num_nei = 0;

    for (unsigned int k = 0u; k < nLinks;
         k++) {  // loop over potential neighbours

        if (num_nei >= nMaxNei) {
            break;
        }
        const unsigned int edge2_idx = d_edge_links[link_begin + k];

        const gbts_edge4 params2 = d_edge_params[edge2_idx];

        const float tau_ratio = gbts_edge_to_float(params2.x) * uat_2 - 1.0f;
        if (fabsf(tau_ratio) > cut_tau_ratio_max) {  // bad match
            continue;
        }

        const float dPhi = phi_wrap(Phi2 - gbts_edge_to_float(params2.w));
        if (fabsf(dPhi) > cut_dphi_max) {
            continue;
        }

        const float dcurv = curv2 - gbts_edge_to_float(params2.y);
        if (fabsf(dcurv) > cut_dcurv_max) {
            continue;
        }

        d_neighbours[nei_pos + num_nei] = edge2_idx;
        d_reIndexer[edge2_idx] = 1;
        ++num_nei;
    }

    d_num_neighbours[globalIndex] = num_nei;

    if (num_nei != 0) {
        d_reIndexer[globalIndex] = 1;
        vecmem::device_atomic_ref<unsigned int>(nConnectionsCounter)
            .fetch_add(static_cast<unsigned int>(num_nei));
    }
}

}  // namespace traccc::device
