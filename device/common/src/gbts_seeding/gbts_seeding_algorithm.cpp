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

gbts_seeding_algorithm::gbts_seeding_algorithm(
    const gbts_seedfinder_config& cfg, const memory_resource& mr,
    vecmem::copy& copy, std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)), algorithm_base{mr, copy}, m_config{cfg} {}

auto gbts_seeding_algorithm::operator()(
    const edm::spacepoint_collection::const_view& spacepoints,
    const edm::measurement_collection::const_view& measurements) const
    -> output_type {

    const unsigned int nSp = copy().get_size(spacepoints);
    TRACCC_DEBUG("nSp " << nSp);
    if (nSp == 0) {
        TRACCC_WARNING("No spacepoints were found in the event");
        return {0, mr().main};
    }

    // Stage 1: nodes. Transient binning buffers die when make_nodes returns.
    node_making_output nodes = make_nodes(spacepoints, measurements);
    if (nodes.nNodes == 0) {
        // No nodes survived spacepoint counting -> no seeds.
        return {0, mr().main};
    }

    // Named counters shared by the graph-making and seed-extraction stages.
    collection_types<unsigned int>::buffer counters_buf(gbts_counter::nCounters,
                                                        mr().main);
    copy().setup(counters_buf)->ignore();
    copy().memset(counters_buf, 0)->ignore();
    collection_types<unsigned int>::host h_counters(
        gbts_counter::nCounters, mr().host ? mr().host : &(mr().main));

    // Stage 2: graph. The per-node buffers are moved in so they are released
    // when make_graph returns, along with all the edge/link transients.
    graph_making_output graph =
        make_graph(std::move(nodes.node_params), std::move(nodes.node_phi),
                   std::move(nodes.node_index), nodes.bin_rads,
                   nodes.eta_bin_views, nodes.nNodes, counters_buf, h_counters);
    if (graph.nConnectedEdges == 0) {
        // No connected edges survived graph making -> no seeds.
        return {0, mr().main};
    }

    // Stage 3: seed extraction.
    return extract_seeds(graph.output_graph, nodes.reducedSP,
                         graph.nConnectedEdges, nSp, counters_buf, h_counters);
}

}  // namespace traccc::device
