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

// Stage 3: seed extraction
auto gbts_seeding_algorithm::extract_seeds(
    collection_types<unsigned int>::buffer& output_graph,
    collection_types<float4>::buffer& reducedSP,
    const unsigned int nConnectedEdges, const unsigned int nSp,
    collection_types<unsigned int>::buffer& counters_buf,
    collection_types<unsigned int>::host& h_counters) const
    -> edm::seed_collection::buffer {

    const gbts_seedfinder_config& cfg = m_config;
    unsigned int* d_counters = counters_buf.ptr();

    // 6. Find longest segments with CCA.
    // active_edges is the per-edge "next iter index" flag: it holds `iter`
    // while the edge is active in iteration `iter`, and -1 once it settles.
    // Iteration 0 writes every entry before any later iteration reads it, so
    // no initialisation is required.
    collection_types<char>::buffer active_edges_buf(nConnectedEdges, mr().main);
    copy().setup(active_edges_buf)->ignore();

    collection_types<unsigned char>::buffer levels_buf(2 * nConnectedEdges,
                                                       mr().main);
    copy().setup(levels_buf)->ignore();
    // Initialise to 1 so a level counts the maximum number of edge segments
    // for a seed originating at the edge.
    copy().memset(levels_buf, 0x1)->ignore();

    collection_types<short2>::buffer outgoing_paths_buf(nConnectedEdges,
                                                        mr().main);
    copy().setup(outgoing_paths_buf)->ignore();

    for (unsigned char iter = 0;
         iter < traccc::device::gbts_consts::max_cca_iter; ++iter) {
        cca_iteration_kernel({nConnectedEdges, cfg.max_num_neighbours,
                              cfg.minLevel, output_graph, levels_buf,
                              active_edges_buf, outgoing_paths_buf, iter});
    }

    count_terminus_edges_kernel({nConnectedEdges, outgoing_paths_buf,
                                 d_counters + gbts_counter::nPaths,
                                 d_counters + gbts_counter::nTerminusEdges});

    copy()(counters_buf, h_counters)->wait();

    const unsigned int nPaths = h_counters[gbts_counter::nPaths];
    const unsigned int nTerminusEdges =
        h_counters[gbts_counter::nTerminusEdges];
    if (nTerminusEdges == 0) {
        TRACCC_WARNING("No terminus edges were found");
        return {0, mr().main};
    }

    TRACCC_DEBUG(nPaths << " size of path store | nTerminusEdges "
                        << nTerminusEdges);

    collection_types<int2>::buffer path_store_buf(nPaths + nTerminusEdges,
                                                  mr().main);
    copy().setup(path_store_buf)->ignore();
    collection_types<int2>::buffer seed_proposals_buf(nPaths, mr().main);
    copy().setup(seed_proposals_buf)->ignore();
    collection_types<char>::buffer seed_ambiguity_buf(nPaths, mr().main);
    copy().setup(seed_ambiguity_buf)->ignore();

    collection_types<unsigned long long int>::buffer edge_bids_buf(
        nConnectedEdges, mr().main);
    copy().setup(edge_bids_buf)->ignore();
    copy().memset(edge_bids_buf, 0)->ignore();

    add_terminus_to_path_store_kernel(
        {nConnectedEdges, path_store_buf, outgoing_paths_buf});

    fill_path_store_kernel({nTerminusEdges, cfg.max_num_neighbours, nPaths,
                            path_store_buf, output_graph, levels_buf,
                            d_counters + gbts_counter::nTerminusEdges});

    fit_segments_kernel({nPaths, nTerminusEdges, cfg.max_num_neighbours,
                         cfg.minLevel, reducedSP, output_graph, path_store_buf,
                         seed_proposals_buf, edge_bids_buf, seed_ambiguity_buf,
                         d_counters + gbts_counter::nTerminusEdges,
                         d_counters + gbts_counter::nProps,
                         cfg.seed_extraction_params, cfg.edge_making.max_z0});

    copy()(counters_buf, h_counters)->wait();

    const unsigned int nProps = h_counters[gbts_counter::nProps];
    TRACCC_DEBUG("nProps " << nProps);
    if (nProps == 0) {
        TRACCC_WARNING("No seed proposals were found");
        return {0, mr().main};
    }

    // 7. Disambiguate seeds through repeated seed-vs-edge bidding rounds.
    for (unsigned int round = 0; round < cfg.edge_bidding_rounds; ++round) {
        copy().memset(edge_bids_buf, 0)->ignore();

        seeds_rebid_for_edges_kernel(
            {nProps, path_store_buf, seed_proposals_buf, edge_bids_buf,
             seed_ambiguity_buf, d_counters + gbts_counter::nRejected,
             round == 0u});

        reset_edge_bids_kernel({nProps, path_store_buf, seed_proposals_buf,
                                edge_bids_buf, seed_ambiguity_buf,
                                d_counters + gbts_counter::nRejected});
    }

    copy()(counters_buf, h_counters)->wait();
    const unsigned int nRejectedProps = h_counters[gbts_counter::nRejected];
    const unsigned int nSeeds =
        (nRejectedProps >= nProps) ? 0u : nProps - nRejectedProps;

    TRACCC_DEBUG("Rejected " << nRejectedProps << " out of " << nProps
                             << " seed proposals");
    if (nSeeds == 0) {
        TRACCC_WARNING("All seed proposals were rejected");
        return {0, mr().main};
    }

    // 8. Convert to 3sp seeds and make output buffer.
    edm::seed_collection::buffer output_seeds(
        2 * nSeeds, mr().main, vecmem::data::buffer_type::resizable);
    copy().setup(output_seeds)->wait();

    collection_types<unsigned long long int>::buffer hit_bids_buf(nSp,
                                                                  mr().main);
    copy().setup(hit_bids_buf)->ignore();
    copy().memset(hit_bids_buf, 0)->wait();

    const unsigned int edge_size = 1u + 2u + cfg.max_num_neighbours;
    seeds_bid_for_hits_kernel({nProps, nSeeds, edge_size, output_graph,
                               seed_proposals_buf, path_store_buf,
                               seed_ambiguity_buf, hit_bids_buf});

    gbts_seed_conversion_kernel(
        {nProps, nSeeds, cfg.max_num_neighbours, seed_proposals_buf,
         seed_ambiguity_buf, path_store_buf, output_graph, reducedSP,
         output_seeds, hit_bids_buf, cfg.seed_ambi_params});

    const unsigned int outputSeeds = copy().get_size(output_seeds);
    TRACCC_DEBUG("GBTS found " << outputSeeds << " seeds");
    return output_seeds;
}
}  // namespace traccc::device
