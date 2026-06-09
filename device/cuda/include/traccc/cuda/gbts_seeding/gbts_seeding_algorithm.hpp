/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/cuda/utils/algorithm_base.hpp"

// Project include(s).
#include "traccc/gbts_seeding/device/gbts_seeding_algorithm.hpp"

namespace traccc::cuda {

/// @brief Main algorithm for performing GBTS seeding on an NVIDIA GPU.
///
/// This algorithm returns a buffer which is not necessarily filled yet. A
/// synchronisation statement is required before destroying this buffer.
///
class gbts_seeding_algorithm : public device::gbts_seeding_algorithm,
                               public cuda::algorithm_base {

    public:
    /// Constructor for the GBTS seed finding algorithm
    ///
    /// @param cfg The GBTS seed finding configuration
    /// @param mr The memory resource(s) to use in the algorithm
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param str The CUDA stream to perform the operations in
    /// @param logger The logger instance to use
    ///
    gbts_seeding_algorithm(
        const gbts_seedfinder_config& cfg, const memory_resource& mr,
        vecmem::copy& copy, cuda::stream& str,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());

    private:
    /// @name Function(s) inherited from @c
    /// traccc::device::gbts_seeding_algorithm
    /// @{

    void count_sp_by_layer_kernel(
        const count_sp_by_layer_kernel_payload& payload) const override;
    void bin_sp_kernel(const bin_sp_kernel_payload& payload) const override;
    void eta_phi_counting_kernel(
        const eta_phi_counting_kernel_payload& payload) const override;
    void eta_phi_prefix_sum_kernel(
        const eta_phi_prefix_sum_kernel_payload& payload) const override;
    void node_sorting_kernel(
        const node_sorting_kernel_payload& payload) const override;
    void minmax_rad_kernel(
        const minmax_rad_kernel_payload& payload) const override;
    void graph_edge_making_kernel(
        const graph_edge_making_kernel_payload& payload) const override;
    void graph_edge_linking_kernel(
        const graph_edge_linking_kernel_payload& payload) const override;
    void graph_edge_matching_kernel(
        const graph_edge_matching_kernel_payload& payload) const override;
    void edge_re_indexing_kernel(
        const edge_re_indexing_kernel_payload& payload) const override;
    void graph_compression_kernel(
        const graph_compression_kernel_payload& payload) const override;
    void cca_iteration_kernel(
        const cca_iteration_kernel_payload& payload) const override;
    void count_terminus_edges_kernel(
        const count_terminus_edges_kernel_payload& payload) const override;
    void add_terminus_to_path_store_kernel(
        const add_terminus_to_path_store_kernel_payload& payload)
        const override;
    void fill_path_store_kernel(
        const fill_path_store_kernel_payload& payload) const override;
    void fit_segments_kernel(
        const fit_segments_kernel_payload& payload) const override;
    void reset_edge_bids_kernel(
        const reset_edge_bids_kernel_payload& payload) const override;
    void seeds_rebid_for_edges_kernel(
        const seeds_rebid_for_edges_kernel_payload& payload) const override;
    void seeds_bid_for_hits_kernel(
        const seeds_bid_for_hits_kernel_payload& payload) const override;
    void gbts_seed_conversion_kernel(
        const gbts_seed_conversion_kernel_payload& payload) const override;

    /// @}

};  // class gbts_seeding_algorithm

}  // namespace traccc::cuda
