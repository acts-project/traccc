/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/algorithm_base.hpp"

// Project include(s).
#include "traccc/edm/container.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/seed_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/vector.hpp>

// System include(s).
#include <cstdint>
#include <memory>
#include <utility>

namespace traccc::device {

/// @brief Main algorithm for performing GBTS seeding on a device
/// (backend-agnostic).
///
/// The algorithm orchestrates the sequence of kernel launches and host-side
/// synchronisations.  Backend-specific subclasses are responsible for
/// implementing the individual kernel launchers.
///
/// This algorithm returns a buffer which is not necessarily filled yet. A
/// synchronisation statement is required before destroying this buffer.
///
class gbts_seeding_algorithm
    : public algorithm<edm::seed_collection::buffer(
          const edm::spacepoint_collection::const_view&,
          const edm::measurement_collection::const_view&)>,
      public messaging,
      public algorithm_base {

    public:
    /// Constructor for the GBTS seed finding algorithm
    ///
    /// @param cfg The GBTS seed finding configuration
    /// @param mr The memory resource(s) to use in the algorithm
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param logger The logger instance to use
    ///
    gbts_seeding_algorithm(
        const gbts_seedfinder_config& cfg, const memory_resource& mr,
        vecmem::copy& copy,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());

    /// Destructor
    virtual ~gbts_seeding_algorithm() = default;

    /// Operator executing the algorithm.
    ///
    /// @param spacepoints is a view of all spacepoints in the event
    /// @param measurements is a view of all measurements in the event
    /// @return the buffer of track seeds reconstructed from the spacepoints
    ///
    output_type operator()(
        const edm::spacepoint_collection::const_view& spacepoints,
        const edm::measurement_collection::const_view& measurements)
        const override;

    protected:
    /// @name Payloads & kernel launchers (to be implemented by backends)
    /// @{

    /// Payload for the count_sp_by_layer_kernel function
    struct count_sp_by_layer_kernel_payload {
        /// Number of spacepoints in the event
        const unsigned int nSp;
        /// All spacepoints in the event
        const edm::spacepoint_collection::const_view& spacepoints;
        /// All measurements in the event (used to look up surface IDs)
        const edm::measurement_collection::const_view& measurements;
        /// Map from detector volume index to GBTS layer index
        const collection_types<short>::const_view& volumeToLayerMap;
        /// Map from (volume, surface) pair to GBTS layer index (optional)
        const collection_types<std::pair<unsigned int, unsigned int>>::
            const_view& surfaceToLayerMap;
        /// Per-layer type code (barrel/endcap/etc.) used for cluster-width cuts
        const collection_types<char>::const_view& layerType;
        /// Output: reduced (x, y, z, r) per spacepoint after filtering
        const collection_types<float4>::view& reducedSP;
        /// Output: per-layer spacepoint counts (atomically incremented)
        const collection_types<unsigned int>::view& layerCounts;
        /// Output: GBTS layer index assigned to each kept spacepoint
        const collection_types<unsigned short>::view& spacepointsLayer;
        /// Size of the volume-to-layer map (for bounds checking)
        const unsigned long int volumeMapSize;
        /// Size of the surface-to-layer map (for bounds checking)
        const unsigned long int surfaceMapSize;
        /// Parameters for SP counting (passed through from config, used for tau
        /// cut if enabled)
        const gbts_sp_counting_params sp_counting_params;
    };

    /// Spacepoint-by-layer counting kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void count_sp_by_layer_kernel(
        const count_sp_by_layer_kernel_payload& payload) const = 0;

    /// Payload for the bin_sp_kernel function
    ///
    /// Each spacepoint is read once and scattered into its layer slot, its
    /// eta/phi bin indices are computed, and the (eta, phi) histogram is
    /// bumped, all in a single pass.
    struct bin_sp_kernel_payload {
        /// Number of spacepoints in the event
        const unsigned int nSp;
        /// Number of phi bins per eta slice
        const unsigned int nPhiBins;
        /// Output: per-spacepoint (x, y, z, r) bin parameters, layer-ordered
        const collection_types<float4>::view& sp_params;
        /// Input: reduced (x, y, z, r) per spacepoint from count_sp_by_layer
        const collection_types<float4>::const_view& reducedSP;
        /// In/out: per-layer running write cursors (decremented as each SP
        /// lands)
        const collection_types<unsigned int>::view& layerCounts;
        /// GBTS layer assignment for each spacepoint
        const collection_types<unsigned short>::const_view& spacepointsLayer;
        /// Output: layer-ordered index back to the original spacepoint slot
        const collection_types<unsigned int>::view& original_sp_idx;
        /// Per-layer (first eta bin, number of eta bins) pair
        const collection_types<
            std::pair<unsigned int, unsigned int>>::const_view& layer_info;
        /// Per-layer geometry pair used to compute eta (e.g. (rmin, zmax))
        const collection_types<std::pair<float, float>>::const_view& layer_geo;
        /// Output: global eta-bin index assigned to each node
        const collection_types<unsigned int>::view& node_eta_index;
        /// Output: phi-bin index assigned to each node
        const collection_types<unsigned int>::view& node_phi_index;
        /// Output: flat (eta, phi) histogram, atomically incremented per node
        const collection_types<unsigned int>::view& eta_phi_histo;
    };

    /// Spacepoint-binning kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void bin_sp_kernel(const bin_sp_kernel_payload& payload) const = 0;

    /// Payload for the eta_phi_counting_kernel function
    struct eta_phi_counting_kernel_payload {
        /// Number of eta bins
        const unsigned int nEtaBins;
        /// Number of phi bins per eta slice
        const unsigned int nPhiBins;
        /// (eta, phi) histogram of node counts
        const collection_types<unsigned int>::const_view& eta_phi_histo;
        /// Output: per-eta total node count (sum over phi)
        const collection_types<unsigned int>::view& eta_node_counter;
        /// Output: per-eta phi prefix-sum scratch (in/out for the next kernel)
        const collection_types<unsigned int>::view& phi_cusums;
    };

    /// Eta-phi counting kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void eta_phi_counting_kernel(
        const eta_phi_counting_kernel_payload& payload) const = 0;

    /// Payload for the eta_phi_prefix_sum_kernel function
    struct eta_phi_prefix_sum_kernel_payload {
        /// Number of eta bins
        const unsigned int nEtaBins;
        /// Number of phi bins per eta slice
        const unsigned int nPhiBins;
        /// Per-eta prefix-summed offsets into the global node array
        const collection_types<unsigned int>::const_view& eta_node_counter;
        /// In/out: per-eta phi prefix sums, made cumulative within each eta
        const collection_types<unsigned int>::view& phi_cusums;
    };

    /// Eta-phi prefix-sum kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void eta_phi_prefix_sum_kernel(
        const eta_phi_prefix_sum_kernel_payload& payload) const = 0;

    /// Payload for the node_sorting_kernel function
    struct node_sorting_kernel_payload {
        /// Total number of GBTS nodes
        const unsigned int nNodes;
        /// Number of phi bins per eta slice
        const unsigned int nPhiBins;
        /// Layer-ordered (x, y, z, r) per spacepoint
        const collection_types<float4>::const_view& sp_params;
        /// Eta-bin index per node
        const collection_types<unsigned int>::const_view& node_eta_index;
        /// Phi-bin index per node
        const collection_types<unsigned int>::const_view& node_phi_index;
        /// In/out: per-(eta, phi) write cursor (atomically advanced)
        const collection_types<unsigned int>::view& phi_cusums;
        /// Output: per-node (tau_min, tau_max, r, z), written in sorted order
        const collection_types<float4>::view& node_params;
        /// Output: per-node phi, written in sorted order
        const collection_types<float>::view& node_phi;
        /// Output: per-sorted-slot original layer-ordered spacepoint index
        const collection_types<unsigned int>::view& node_index;
        /// Map from layer-ordered SP index to the original SP slot
        const collection_types<unsigned int>::const_view& original_sp_idx;
        /// Optional tau lookup table (used iff node_sorting_params.useTauLUT)
        const collection_types<float>::const_view& tau_lut;
        /// Tau-prediction cuts read by @c device::node_sorting
        const gbts_node_sorting_params node_sorting_params;
    };

    /// Node sorting kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void node_sorting_kernel(
        const node_sorting_kernel_payload& payload) const = 0;

    /// Payload for the minmax_rad_kernel function
    struct minmax_rad_kernel_payload {
        /// Number of eta bins
        const unsigned int nEtaBins;
        /// Per-eta (begin, end) node range, as 2*nEtaBins flat ints
        const collection_types<unsigned int>::const_view& eta_bin_views;
        /// Per-node (tau_min, tau_max, r, z) (only r is read here)
        const collection_types<float4>::const_view& node_params;
        /// Output: per-eta (rmin, rmax) pair, flat (2*nEtaBins floats)
        const collection_types<float>::view& bin_rads;
    };

    /// Min/max radius per eta-bin kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void minmax_rad_kernel(
        const minmax_rad_kernel_payload& payload) const = 0;

    /// Payload for the graph_edge_making_kernel function
    struct graph_edge_making_kernel_payload {
        /// Number of bin-pair tasks (also the CUDA block count)
        const unsigned int nUsedBinPairs;
        /// Upper bound on the number of edges to write
        const unsigned int nMaxEdges;
        /// Number of phi bins per eta slice
        const unsigned int nPhiBins;
        /// Per-bin-pair (begin1, end1, begin2, end2) node ranges, flat
        const collection_types<unsigned int>::const_view& bin_pair_views;
        /// Per-bin-pair max delta-phi window for edge candidates
        const collection_types<float>::const_view& bin_pair_dphi;
        /// Per-node (tau_min, tau_max, r, z)
        const collection_types<float4>::const_view& node_params;
        /// Per-node phi
        const collection_types<float>::const_view& node_phi;
        /// Edge-making geometric / kinematic cuts
        const gbts_edge_making_params edge_making_params;
        /// In/out: global atomic counter for the next edge slot to write
        unsigned int* nEdgesCounter;
        /// Output: (src, dst) node indices per edge
        const collection_types<uint2>::view& edge_nodes;
        /// Output: packed per-edge [exp(-eta), curv, phi_z, phi_w] used by
        /// matching
        const collection_types<gbts_edge4>::view& edge_params;
        /// Output: per-destination-node incoming-edge count (atomic)
        const collection_types<unsigned int>::view& num_outgoing_edges;
    };

    /// Graph edge-making kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void graph_edge_making_kernel(
        const graph_edge_making_kernel_payload& payload) const = 0;

    /// Payload for the graph_edge_linking_kernel function
    struct graph_edge_linking_kernel_payload {
        /// Number of edges produced earlier
        const unsigned int nEdges;
        /// (src, dst) node indices per edge
        const collection_types<uint2>::const_view& edge_nodes;
        /// Output: per-edge slot in the per-node incoming-edge list
        const collection_types<unsigned int>::view& edge_links;
        /// In/out: per-node prefix-sum / write cursor of incoming edges
        const collection_types<unsigned int>::view& num_outgoing_edges;
    };

    /// Graph edge-linking kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void graph_edge_linking_kernel(
        const graph_edge_linking_kernel_payload& payload) const = 0;

    /// Payload for the graph_edge_matching_kernel function
    struct graph_edge_matching_kernel_payload {
        /// Number of edges to match
        const unsigned int nEdges;
        /// Maximum number of neighbours retained per edge
        const unsigned int nMaxNei;
        /// Edge-matching pair cuts
        const gbts_edge_matching_params edge_matching_params;
        /// Packed per-edge [exp(-eta), curv, phi_z, phi_w], from
        /// graph_edge_making
        const collection_types<gbts_edge4>::const_view& edge_params;
        /// (src, dst) node indices per edge
        const collection_types<uint2>::const_view& edge_nodes;
        /// Per-node prefix sum of incoming edges (used to locate candidates)
        const collection_types<unsigned int>::const_view& num_outgoing_edges;
        /// Per-edge slot in its destination node's incoming-edge list
        const collection_types<unsigned int>::const_view& edge_links;
        /// Output: number of accepted neighbours per edge (0..nMaxNei)
        const collection_types<unsigned char>::view& num_neighbours;
        /// Output: neighbour edge indices, nMaxNei entries per edge (flat)
        const collection_types<unsigned int>::view& neighbours;
        /// Output: per-edge "kept" flag, later compacted into a re-index
        const collection_types<int>::view& reIndexer;
        /// In/out: global atomic counter of total accepted connections
        unsigned int* nConnectionsCounter;
    };

    /// Graph edge-matching kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void graph_edge_matching_kernel(
        const graph_edge_matching_kernel_payload& payload) const = 0;

    /// Payload for the edge_re_indexing_kernel function
    struct edge_re_indexing_kernel_payload {
        /// Number of original edges
        const unsigned int nEdges;
        /// In/out: per-edge "kept" flag in, compact new index out
        const collection_types<int>::view& reIndexer;
        /// In/out: global atomic counter of edges that survived re-indexing
        unsigned int* nConnectedEdgesCounter;
    };

    /// Edge re-indexing kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void edge_re_indexing_kernel(
        const edge_re_indexing_kernel_payload& payload) const = 0;

    /// Payload for the graph_compression_kernel function
    struct graph_compression_kernel_payload {
        /// Number of original (uncompressed) edges
        const unsigned int nEdges;
        /// Maximum number of neighbours retained per edge
        const unsigned int nMaxNei;
        /// Sorted-slot to original spacepoint index map
        const collection_types<unsigned int>::const_view& orig_node_index;
        /// (src, dst) node indices per edge
        const collection_types<uint2>::const_view& edge_nodes;
        /// Number of accepted neighbours per edge
        const collection_types<unsigned char>::const_view& num_neighbours;
        /// Neighbour edge indices per edge (nMaxNei per edge, flat)
        const collection_types<unsigned int>::const_view& neighbours;
        /// Old-edge to compacted-edge index map
        const collection_types<int>::const_view& reIndexer;
        /// Output: compacted graph in row-major layout; each edge owns a block
        /// of edge_size = 2 + 1 + nMaxNei ints (node1, node2, nNei,
        /// nei0..neiN-1).
        const collection_types<unsigned int>::view& output_graph;
    };

    /// Graph compression kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void graph_compression_kernel(
        const graph_compression_kernel_payload& payload) const = 0;

    /// Payload for the cca_iteration_kernel function
    struct cca_iteration_kernel_payload {
        /// Number of edges in the compacted graph
        const unsigned int nConnectedEdges;
        /// Maximum number of neighbours retained per edge
        const unsigned int max_num_neighbours;
        /// Minimum path length required for an edge to be considered active
        const unsigned char minLevel;
        /// Compacted graph from graph_compression
        const collection_types<unsigned int>::const_view& output_graph;
        /// In/out: per-edge level ping-pong buffer (2 * nConnectedEdges bytes)
        const collection_types<unsigned char>::view& levels;
        /// In/out: per-edge active-flag (holds the next iter index, or -1
        /// once the edge is no longer active).
        const collection_types<char>::view& active_edges;
        /// Output: longest outgoing-path summary per edge (length, next-edge)
        const collection_types<short2>::view& outgoing_paths;
        // Iteration index (0-based)
        const unsigned char iter;
    };

    /// CCA (connected-components iteration) kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void cca_iteration_kernel(
        const cca_iteration_kernel_payload& payload) const = 0;

    /// Payload for the count_terminus_edges_kernel function
    struct count_terminus_edges_kernel_payload {
        /// Number of edges in the compacted graph
        const unsigned int nConnectedEdges;
        /// Per-edge longest-outgoing-path summary from CCA
        const collection_types<short2>::view& outgoing_paths;
        /// Total number of paths reachable from any terminus edge
        unsigned int* nPathsCounter;
        /// Running size of the path store (initialised to nTerminusEdges)
        unsigned int* nPathStoreSizeCounter;
    };

    /// Terminus-edge counting kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void count_terminus_edges_kernel(
        const count_terminus_edges_kernel_payload& payload) const = 0;

    /// Payload for the add_terminus_to_path_store_kernel function
    struct add_terminus_to_path_store_kernel_payload {
        /// Number of edges in the compacted graph
        const unsigned int nConnectedEdges;
        /// Output: per-path (edge index, parent path-store index or -1)
        /// entries; terminus rows occupy the first nTerminusEdges slots
        const collection_types<int2>::view& path_store;
        /// Per-edge longest-outgoing-path summary from CCA
        const collection_types<short2>::const_view& outgoing_paths;
    };

    /// Terminus-to-path-store seeding kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void add_terminus_to_path_store_kernel(
        const add_terminus_to_path_store_kernel_payload& payload) const = 0;

    /// Payload for the fill_path_store_kernel function
    struct fill_path_store_kernel_payload {
        /// Number of terminus edges
        const unsigned int nTerminusEdges;
        /// Maximum number of neighbours retained per edge
        const unsigned int max_num_neighbours;
        /// Total number of paths
        const unsigned int nPaths;
        /// In/out: per-path (edge index, parent path-store index or -1) entries
        const collection_types<int2>::view& path_store;
        /// Compacted graph (read for per-edge neighbour lookup)
        const collection_types<unsigned int>::const_view& output_graph;
        /// Per-edge CCA level array
        const collection_types<unsigned char>::const_view& levels;
        /// In/out: global atomic write cursor into path_store
        unsigned int* nPathStoreSizeCounter;
    };

    /// Path-store-filling kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void fill_path_store_kernel(
        const fill_path_store_kernel_payload& payload) const = 0;

    /// Payload for the fit_segments_kernel function
    struct fit_segments_kernel_payload {
        /// Upper bound on the number of paths (== path_store size minus
        /// terminus prefix)
        const unsigned int nPaths;
        /// Number of terminus edges (path-store offset for fittable paths)
        const unsigned int nTerminusEdges;
        /// Maximum number of neighbours retained per edge
        const unsigned int max_num_neighbours;
        /// Minimum number of edges a path must have to be fit
        const unsigned char minLevel;
        /// Reduced (x, y, z, r) per original spacepoint
        const collection_types<float4>::const_view& reducedSP;
        /// Compacted graph from graph_compression
        const collection_types<unsigned int>::const_view& output_graph;
        /// Per-path (edge index, parent path-store index or -1) entries
        const collection_types<int2>::const_view& path_store;
        /// Output: per-accepted-path (path_store index, level) seed proposal
        const collection_types<int2>::view& seed_proposals;
        /// In/out: per-edge highest-bidder seed proposal (packed 64-bit)
        const collection_types<unsigned long long int>::view& edge_bids;
        /// Output: per-seed-proposal ambiguity tag (multi-bid resolution flag)
        const collection_types<char>::view& seed_ambiguity;
        /// Read-only upper bound on path indices (set by earlier kernels)
        unsigned int* nPathStoreSize;
        /// In/out: global atomic counter of accepted seed proposals
        unsigned int* nPropsCounter;
        /// Curvature / pT / chi-squared cut parameters
        const gbts_seed_extraction_params seed_extraction_params;
        /// Maximum |z0| at the beamline for extrapolation cuts
        const float max_z0;
    };

    /// Segment fitting kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void fit_segments_kernel(
        const fit_segments_kernel_payload& payload) const = 0;

    /// Payload for the reset_edge_bids_kernel function
    struct reset_edge_bids_kernel_payload {
        /// Number of seed proposals
        const unsigned int nProps;
        /// Per-path (edge index, parent path-store index or -1) entries
        const collection_types<int2>::const_view& path_store;
        /// In/out: per-seed-proposal (path_store index, level)
        const collection_types<int2>::view& seed_proposals;
        /// In/out: per-edge highest-bidder seed proposal (cleared between
        /// rounds)
        const collection_types<unsigned long long int>::view& edge_bids;
        /// In/out: per-seed-proposal ambiguity tag
        const collection_types<char>::view& seed_ambiguity;
        /// In/out: global atomic counter of rejected proposals
        unsigned int* nRejectedPropsCounter;
    };

    /// Edge-bid reset kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void reset_edge_bids_kernel(
        const reset_edge_bids_kernel_payload& payload) const = 0;

    /// Payload for the seeds_rebid_for_edges_kernel function
    struct seeds_rebid_for_edges_kernel_payload {
        /// Number of seed proposals
        const unsigned int nProps;
        /// Per-path (edge index, parent path-store index or -1) entries
        const collection_types<int2>::const_view& path_store;
        /// Per-seed-proposal (path_store index, level)
        const collection_types<int2>::view& seed_proposals;
        /// In/out: per-edge highest-bidder seed proposal (cleared on entry)
        const collection_types<unsigned long long int>::view& edge_bids;
        /// In/out: per-seed-proposal ambiguity tag
        const collection_types<char>::view& seed_ambiguity;
        /// In/out: global atomic counter of rejected proposals
        unsigned int* nRejectedPropsCounter;
        /// True on the first bidding round (folds the init pass)
        const bool first_round;
    };

    /// Edge re-bid kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void seeds_rebid_for_edges_kernel(
        const seeds_rebid_for_edges_kernel_payload& payload) const = 0;

    /// Payload for the seeds_bid_for_hits_kernel function
    struct seeds_bid_for_hits_kernel_payload {
        /// Number of seed proposals
        const unsigned int nProps;
        /// Number of accepted seeds (nProps - nRejectedProps)
        const unsigned int nSeeds;
        /// Per-edge row stride in the output graph (= 2 + 1 +
        /// max_num_neighbours)
        const unsigned int edge_size;
        /// Compacted graph from graph_compression
        const collection_types<unsigned int>::const_view& output_graph;
        /// Per-seed-proposal (path_store index, level)
        const collection_types<int2>::const_view& seed_proposals;
        /// Per-path (edge index, parent path-store index or -1) entries
        const collection_types<int2>::const_view& path_store;
        /// Per-seed-proposal ambiguity tag
        const collection_types<char>::const_view& seed_ambiguity;
        /// In/out: per-hit highest-bidder seed (packed 64-bit)
        const collection_types<unsigned long long int>::view& hit_bids;
    };

    /// Seeds-bid-for-hits kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void seeds_bid_for_hits_kernel(
        const seeds_bid_for_hits_kernel_payload& payload) const = 0;

    /// Payload for the gbts_seed_conversion_kernel function
    struct gbts_seed_conversion_kernel_payload {
        /// Number of seed proposals
        const unsigned int nProps;
        /// Number of accepted seeds (nProps - nRejectedProps)
        const unsigned int nSeeds;
        /// Maximum number of neighbours retained per edge
        const unsigned int max_num_neighbours;
        /// Per-seed-proposal (path_store index, level)
        const collection_types<int2>::const_view& seed_proposals;
        /// Per-seed-proposal ambiguity tag
        const collection_types<char>::const_view& seed_ambiguity;
        /// Per-path (edge index, parent path-store index or -1) entries
        const collection_types<int2>::const_view& path_store;
        /// Compacted graph from graph_compression
        const collection_types<unsigned int>::const_view& output_graph;
        /// Reduced (x, y, z, r) per original spacepoint
        const collection_types<float4>::const_view& reducedSP;
        /// Output: 3-SP seeds appended to this resizable buffer
        const edm::seed_collection::view& output_seeds;
        /// Per-hit highest-bidder seed (read for dropout decisions)
        const collection_types<unsigned long long int>::view& hit_bids;
        /// Dropout / curvature / ambiguity cut parameters
        const gbts_seed_ambi_params seed_ambi_params;
    };

    /// GBTS seed conversion kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void gbts_seed_conversion_kernel(
        const gbts_seed_conversion_kernel_payload& payload) const = 0;

    /// @}

    private:
    /// @name Pipeline stages
    ///
    /// The pipeline is split into three stage methods so that each stage's
    /// transient device buffers are local and freed as soon as the stage
    /// returns; only the cross-stage handles below survive between stages.
    /// @{

    /// Outputs of the node-making stage that are consumed downstream.
    struct node_making_output {
        /// Reduced (x, y, z, r) per original spacepoint (used by seed
        /// extraction)
        collection_types<float4>::buffer reducedSP;
        /// Per-node (tau_min, tau_max, r, z) (used by graph making)
        collection_types<float4>::buffer node_params;
        /// Per-node phi (used by graph making)
        collection_types<float>::buffer node_phi;
        /// Per-sorted-slot original spacepoint index (used by graph making)
        collection_types<unsigned int>::buffer node_index;
        /// Per-eta (rmin, rmax) pair, host (used by graph making)
        collection_types<float>::host bin_rads;
        /// Per-eta (begin, end) node ranges, host (used by graph making)
        collection_types<unsigned int>::host eta_bin_views;
        /// Number of GBTS nodes (0 == nothing to do)
        unsigned int nNodes = 0;
    };

    /// Outputs of the graph-making stage that are consumed by seed extraction.
    struct graph_making_output {
        /// Compacted, row-major graph
        collection_types<unsigned int>::buffer output_graph;
        /// Number of edges that survived re-indexing (0 == nothing to do)
        unsigned int nConnectedEdges = 0;
    };

    /// Stage 1: count, bin, sort and characterise nodes.
    node_making_output make_nodes(
        const edm::spacepoint_collection::const_view& spacepoints,
        const edm::measurement_collection::const_view& measurements) const;

    /// Stage 2: build, link, match and compress the edge graph. The per-node
    /// buffers are taken by value so they are released when this stage returns.
    graph_making_output make_graph(
        collection_types<float4>::buffer node_params,
        collection_types<float>::buffer node_phi,
        collection_types<unsigned int>::buffer node_index,
        const collection_types<float>::host& bin_rads,
        const collection_types<unsigned int>::host& eta_bin_views,
        const unsigned int nNodes,
        collection_types<unsigned int>::buffer& counters_buf,
        collection_types<unsigned int>::host& h_counters) const;

    /// Stage 3: run the CCA, extract paths, fit and disambiguate into seeds.
    edm::seed_collection::buffer extract_seeds(
        collection_types<unsigned int>::buffer& output_graph,
        collection_types<float4>::buffer& reducedSP,
        const unsigned int nConnectedEdges, const unsigned int nSp,
        collection_types<unsigned int>::buffer& counters_buf,
        collection_types<unsigned int>::host& h_counters) const;

    /// @}

    /// GBTS seed-finding configuration.
    gbts_seedfinder_config m_config;

};  // class gbts_seeding_algorithm

}  // namespace traccc::device
