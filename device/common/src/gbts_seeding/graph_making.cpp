/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/gbts_seeding/device/gbts_seeding_algorithm.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/vector.hpp>

// System include(s).
#include <algorithm>
#include <cmath>
#include <utility>

namespace traccc::device {

// Stage 2: graph making
auto gbts_seeding_algorithm::make_graph(
    collection_types<float4>::buffer node_params,
    collection_types<float>::buffer node_phi,
    collection_types<unsigned int>::buffer node_index,
    const collection_types<float>::host& bin_rads,
    const collection_types<unsigned int>::host& eta_bin_views,
    const unsigned int nNodes,
    collection_types<unsigned int>::buffer& counters_buf,
    collection_types<unsigned int>::host& h_counters) const
    -> graph_making_output {

    const gbts_seedfinder_config& cfg = m_config;
    unsigned int* d_counters = counters_buf.ptr();

    // CPU: build the per-bin-pair work list (begin/end node ranges + phi search
    // window) on the host from the eta-bin views, splitting large bins into
    // node_buffer_length-sized chunks. Two passes: count then fill.
    int int_nBinPairs = 0;
    for (const std::pair<unsigned int, unsigned int>& binPair : cfg.binTables) {
        const int bin1_begin = eta_bin_views[2 * binPair.first];
        const int bin1_end = eta_bin_views[2 * binPair.first + 1];
        int nNodesInBin1 = bin1_end - bin1_begin;
        if (bin1_begin > bin1_end) {
            nNodesInBin1 = bin1_begin - bin1_end;
        }
        int_nBinPairs +=
            1 + (nNodesInBin1 - 1) / gbts_consts::node_buffer_length;
    }
    const unsigned int nBinPairs = static_cast<unsigned int>(int_nBinPairs);

    collection_types<unsigned int>::host bin_pair_views(4 * nBinPairs,
                                                        mr().host);
    collection_types<float>::host bin_pair_dphi(nBinPairs, mr().host);

    unsigned int pairIdx = 0;
    for (const std::pair<unsigned int, unsigned int>& binPair : cfg.binTables) {
        const float rb1 = bin_rads[2 * binPair.first];

        const unsigned int begin_bin1 = eta_bin_views[2 * binPair.first];
        const unsigned int end_bin1 = eta_bin_views[2 * binPair.first + 1];
        if (begin_bin1 == end_bin1) {
            continue;
        }
        if (eta_bin_views[2 * binPair.second] ==
            eta_bin_views[2 * binPair.second + 1]) {
            continue;
        }

        const float rb2 = bin_rads[2 * binPair.second + 1];
        const float maxDeltaR = std::fabs(rb2 - rb1);

        float deltaPhi = cfg.dphi_window.min_delta_phi +
                         cfg.dphi_window.dphi_coeff * maxDeltaR;
        if (maxDeltaR < cfg.dphi_window.low_dr_threshold) {
            deltaPhi = cfg.dphi_window.min_delta_phi_low_dr +
                       cfg.dphi_window.dphi_coeff_low_dr * maxDeltaR;
        }

        unsigned int currBegin_bin1 = begin_bin1;
        unsigned int currEnd_bin1 =
            end_bin1 < gbts_consts::node_buffer_length
                ? end_bin1
                : begin_bin1 + gbts_consts::node_buffer_length;

        for (; currEnd_bin1 < end_bin1;
             currEnd_bin1 += gbts_consts::node_buffer_length, pairIdx++) {
            const unsigned int offset = 4 * pairIdx;
            bin_pair_views[offset] = currBegin_bin1;
            bin_pair_views[1 + offset] = currEnd_bin1;
            bin_pair_views[2 + offset] = eta_bin_views[2 * binPair.second];
            bin_pair_views[3 + offset] = eta_bin_views[2 * binPair.second + 1];
            bin_pair_dphi[pairIdx] = deltaPhi;
            currBegin_bin1 = currEnd_bin1;
        }
        currEnd_bin1 = end_bin1;

        const unsigned int offset = 4 * pairIdx;
        bin_pair_views[offset] = currBegin_bin1;
        bin_pair_views[1 + offset] = currEnd_bin1;
        bin_pair_views[2 + offset] = eta_bin_views[2 * binPair.second];
        bin_pair_views[3 + offset] = eta_bin_views[2 * binPair.second + 1];
        bin_pair_dphi[pairIdx] = deltaPhi;
        pairIdx++;
    }
    const unsigned int nUsedBinPairs = pairIdx;
    TRACCC_DEBUG("nUsedBinPairs " << nUsedBinPairs);
    if (nUsedBinPairs == 0) {
        TRACCC_WARNING("No bin pairs were used for edge finding");
        return graph_making_output{};
    }

    collection_types<unsigned int>::buffer bin_pair_views_buf(4 * nUsedBinPairs,
                                                              mr().main);
    copy().setup(bin_pair_views_buf)->ignore();
    copy()(vecmem::get_data(bin_pair_views), bin_pair_views_buf)->ignore();

    collection_types<float>::buffer bin_pair_dphi_buf(nUsedBinPairs, mr().main);
    copy().setup(bin_pair_dphi_buf)->ignore();
    copy()(vecmem::get_data(bin_pair_dphi), bin_pair_dphi_buf)->ignore();

    // 2. Find edges between spacepoint pairs.
    const unsigned int nMaxEdges = cfg.max_edges_factor * nNodes;
    // Packed per-edge parameter buffer ([exp(-eta), curv, phi_z, phi_w]).
    collection_types<gbts_edge4>::buffer edge_params_buf(nMaxEdges, mr().main);
    copy().setup(edge_params_buf)->ignore();
    collection_types<uint2>::buffer edge_nodes_buf(nMaxEdges, mr().main);
    copy().setup(edge_nodes_buf)->ignore();
    collection_types<unsigned int>::buffer num_incoming_edges_buf(nNodes + 1,
                                                                  mr().main);
    copy().setup(num_incoming_edges_buf)->ignore();
    copy().memset(num_incoming_edges_buf, 0)->ignore();

    graph_edge_making_kernel({nUsedBinPairs, nMaxEdges, cfg.n_phi_bins,
                              bin_pair_views_buf, bin_pair_dphi_buf,
                              node_params, node_phi, cfg.edge_making,
                              d_counters + gbts_counter::nEdges, edge_nodes_buf,
                              edge_params_buf, num_incoming_edges_buf});

    // Read back the number of edges produced.
    copy()(counters_buf, h_counters)->wait();

    unsigned int nEdges = h_counters[gbts_counter::nEdges];
    TRACCC_DEBUG("Created " << nEdges << " edges with a cap of " << nMaxEdges);
    if (nEdges > nMaxEdges) {
        TRACCC_WARNING("Number of edges exceeds the maximum allowed, Removing "
                       << nEdges - nMaxEdges << " edges");
        nEdges = nMaxEdges;
    } else if (nEdges == 0) {
        TRACCC_WARNING("No edges were found");
        return graph_making_output{};
    }

    // CPU: prefix-sum the per-node incoming-edge counts on the host into the
    // write cursors used by graph_edge_linking.
    collection_types<unsigned int>::host cusum(nNodes + 1, mr().host);
    copy()(vecmem::get_data(num_incoming_edges_buf), cusum)->wait();
    for (unsigned int k = 0; k < nNodes; k++) {
        cusum[k + 1] += cusum[k];
    }
    copy()(vecmem::get_data(cusum), num_incoming_edges_buf)->wait();

    // 3. Link edges and nodes.
    collection_types<unsigned int>::buffer edge_links_buf(nEdges, mr().main);
    copy().setup(edge_links_buf)->ignore();

    graph_edge_linking_kernel(
        {nEdges, edge_nodes_buf, edge_links_buf, num_incoming_edges_buf});

    // 4. Edge matching to create edge-to-edge connections.
    collection_types<unsigned char>::buffer num_neighbours_buf(nEdges,
                                                               mr().main);
    copy().setup(num_neighbours_buf)->ignore();
    copy().memset(num_neighbours_buf, 0)->ignore();

    collection_types<int>::buffer reIndexer_buf(nEdges, mr().main);
    copy().setup(reIndexer_buf)->ignore();
    // Byte-fill 0xFF -> int -1, the "edge not kept" sentinel checked by
    // edge_re_indexing / graph_compression.
    copy().memset(reIndexer_buf, 0xFF)->ignore();

    collection_types<unsigned int>::buffer neighbours_buf(
        cfg.max_num_neighbours * nEdges, mr().main);
    copy().setup(neighbours_buf)->ignore();
    copy().memset(neighbours_buf, 0)->ignore();

    graph_edge_matching_kernel(
        {nEdges, cfg.max_num_neighbours, cfg.edge_matching, edge_params_buf,
         edge_nodes_buf, num_incoming_edges_buf, edge_links_buf,
         num_neighbours_buf, neighbours_buf, reIndexer_buf,
         d_counters + gbts_counter::nConnections});

    // 5. Edge re-indexing to keep only edges involved in any connection.
    edge_re_indexing_kernel(
        {nEdges, reIndexer_buf, d_counters + gbts_counter::nConnectedEdges});

    copy()(counters_buf, h_counters)->wait();

    const unsigned int nConnections = h_counters[gbts_counter::nConnections];
    const unsigned int nConnectedEdges =
        h_counters[gbts_counter::nConnectedEdges];
    TRACCC_DEBUG("created " << nConnections << " edge links, found "
                            << nConnectedEdges
                            << " connected edges for seed extraction");
    if (nConnectedEdges == 0) {
        TRACCC_WARNING("No connected edges were found");
        return graph_making_output{};
    }

    const unsigned int nIntsPerEdge = 2 + 1 + cfg.max_num_neighbours;
    collection_types<unsigned int>::buffer output_graph_buf(
        nConnectedEdges * nIntsPerEdge, mr().main);
    copy().setup(output_graph_buf)->ignore();

    graph_compression_kernel({nEdges, cfg.max_num_neighbours, node_index,
                              edge_nodes_buf, num_neighbours_buf,
                              neighbours_buf, reIndexer_buf, output_graph_buf});

    return graph_making_output{std::move(output_graph_buf), nConnectedEdges};
}
}  // namespace traccc::device
