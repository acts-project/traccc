/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/cuda/utils/stream.hpp"

// Project include(s).
#include "traccc/clusterization/device/clusterization_algorithm.hpp"

namespace traccc::cuda {

/// Algorithm performing hit clusterization
///
/// This algorithm implements hit clusterization in a massively-parallel
/// approach. Each thread handles a pre-determined number of detector cells.
///
/// This algorithm returns a buffer which is not necessarily filled yet. A
/// synchronisation statement is required before destroying the buffer.
///
class clusterization_algorithm : public device::clusterization_algorithm {

    public:
    /// Constructor for clusterization algorithm
    ///
    /// @param mr The memory resource(s) to use in the algorithm
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param str The CUDA stream to perform the operations in
    /// @param config The clustering configuration partition
    /// @param logger The logger instance to use for messaging
    ///
    clusterization_algorithm(
        const traccc::memory_resource& mr, vecmem::copy& copy, stream& str,
        const config_type& config,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());

    private:
    /// @name Function(s) inherited from the base class
    /// @{

    /// Function meant to perform sanity checks on the input data
    ///
    /// @param cells     All cells in an event
    /// @return @c true if the input data is valid, @c false otherwise
    ///
    bool input_is_valid(
        const edm::silicon_cell_collection::const_view& cells) const override;

    /// Main CCL kernel launcher
    ///
    /// @param num_cells     Number of cells in the event
    /// @param config        The clustering configuration
    /// @param cells         All cells in an event
    /// @param det_descr     The detector description
    /// @param measurements  The measurement collection to fill
    /// @param cell_links    Buffer for linking cells to measurements
    /// @param f_backup      Buffer for backup of the first element links
    /// @param gf_backup     Buffer for backup of the group first element links
    /// @param adjc_backup   Buffer for backup of the adjacency matrix (counts)
    /// @param adjv_backup   Buffer for backup of the adjacency matrix (values)
    /// @param backup_mutex  Mutex for the backup structures
    /// @param disjoint_set  Buffer for the disjoint set data structure
    /// @param cluster_sizes Buffer for the sizes of the clusters
    ///
    void ccl_kernel(
        unsigned int num_cells, const config_type& config,
        const edm::silicon_cell_collection::const_view& cells,
        const silicon_detector_description::const_view& det_descr,
        edm::measurement_collection<default_algebra>::view& measurements,
        vecmem::data::vector_view<unsigned int>& cell_links,
        vecmem::data::vector_view<device::details::index_t>& f_backup,
        vecmem::data::vector_view<device::details::index_t>& gf_backup,
        vecmem::data::vector_view<unsigned char>& adjc_backup,
        vecmem::data::vector_view<device::details::index_t>& adjv_backup,
        unsigned int* backup_mutex,
        vecmem::data::vector_view<unsigned int>& disjoint_set,
        vecmem::data::vector_view<unsigned int>& cluster_sizes) const override;

    /// Cluster data reification kernel launcher
    ///
    /// @param num_cells    Number of cells in the event
    /// @param disjoint_set Buffer for the disjoint set data structure
    /// @param cluster_data The cluster collection to fill
    ///
    void cluster_maker_kernel(
        unsigned int num_cells,
        const vecmem::data::vector_view<unsigned int>& disjoint_set,
        edm::silicon_cluster_collection::view& cluster_data) const override;

    /// @}

    /// The CUDA stream to use
    std::reference_wrapper<stream> m_stream;

};  // class clusterization_algorithm

}  // namespace traccc::cuda
