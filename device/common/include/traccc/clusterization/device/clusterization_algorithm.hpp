/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/clusterization/device/ccl_kernel_definitions.hpp"
#include "traccc/clusterization/device/tags.hpp"
#include "traccc/device/algorithm_base.hpp"

// Project include(s).
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/edm/silicon_cluster_collection.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

// VecMem include(s).
#include <vecmem/memory/unique_ptr.hpp>
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <functional>
#include <optional>

namespace traccc::device {

/// Base class for the algorithms performing hit clusterization
///
/// This algorithm implements hit clusterization in a massively-parallel
/// approach. Each thread handles a pre-determined number of detector cells.
///
/// This algorithm returns a buffer which is not necessarily filled yet. A
/// synchronisation statement is required before destroying the buffer.
///
class clusterization_algorithm
    : public algorithm<edm::measurement_collection<default_algebra>::buffer(
          const edm::silicon_cell_collection::const_view&,
          const silicon_detector_description::const_view&)>,
      public algorithm<edm::measurement_collection<default_algebra>::buffer(
          const edm::silicon_cell_collection::const_view&,
          const silicon_detector_description::const_view&,
          clustering_discard_disjoint_set&&)>,
      public algorithm<
          std::pair<edm::measurement_collection<default_algebra>::buffer,
                    edm::silicon_cluster_collection::buffer>(
              const edm::silicon_cell_collection::const_view&,
              const silicon_detector_description::const_view&,
              clustering_keep_disjoint_set&&)>,
      public messaging,
      public algorithm_base {

    public:
    /// Configuration type
    using config_type = clustering_config;

    /// Constructor for clusterization algorithm
    ///
    /// @param mr The memory resource(s) to use in the algorithm
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param config The clustering configuration
    /// partition
    ///
    clusterization_algorithm(
        const traccc::memory_resource& mr, vecmem::copy& copy,
        const config_type& config,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());

    /// Callable operator for clusterization algorithm
    ///
    /// @param cells     All cells in an event
    /// @param det_descr The detector description
    /// @return a measurement collection (buffer)
    ///
    /// @{
    edm::measurement_collection<default_algebra>::buffer operator()(
        const edm::silicon_cell_collection::const_view& cells,
        const silicon_detector_description::const_view& det_descr)
        const override;

    edm::measurement_collection<default_algebra>::buffer operator()(
        const edm::silicon_cell_collection::const_view& cells,
        const silicon_detector_description::const_view& det_descr,
        clustering_discard_disjoint_set&&) const override;

    std::pair<edm::measurement_collection<default_algebra>::buffer,
              edm::silicon_cluster_collection::buffer>
    operator()(const edm::silicon_cell_collection::const_view& cells,
               const silicon_detector_description::const_view& det_descr,
               clustering_keep_disjoint_set&&) const override;
    /// @}

    protected:
    /// @name Function(s) to be implemented by derived classes
    /// @{

    /// Function meant to perform sanity checks on the input data
    ///
    /// @param cells     All cells in an event
    /// @return @c true if the input data is valid, @c false otherwise
    ///
    virtual bool input_is_valid(
        const edm::silicon_cell_collection::const_view& cells) const = 0;

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
    virtual void ccl_kernel(
        unsigned int num_cells, const config_type& config,
        const edm::silicon_cell_collection::const_view& cells,
        const silicon_detector_description::const_view& det_descr,
        edm::measurement_collection<default_algebra>::view& measurements,
        vecmem::data::vector_view<unsigned int>& cell_links,
        vecmem::data::vector_view<details::index_t>& f_backup,
        vecmem::data::vector_view<details::index_t>& gf_backup,
        vecmem::data::vector_view<unsigned char>& adjc_backup,
        vecmem::data::vector_view<details::index_t>& adjv_backup,
        unsigned int* backup_mutex,
        vecmem::data::vector_view<unsigned int>& disjoint_set,
        vecmem::data::vector_view<unsigned int>& cluster_sizes) const = 0;

    /// Cluster data reification kernel launcher
    ///
    /// @param num_cells    Number of cells in the event
    /// @param disjoint_set Buffer for the disjoint set data structure
    /// @param cluster_data The cluster collection to fill
    ///
    virtual void cluster_maker_kernel(
        unsigned int num_cells,
        const vecmem::data::vector_view<unsigned int>& disjoint_set,
        edm::silicon_cluster_collection::view& cluster_data) const = 0;

    /// @}

    private:
    /// Main algorithmic implementation of the clusterization algorithm
    std::pair<edm::measurement_collection<default_algebra>::buffer,
              std::optional<edm::silicon_cluster_collection::buffer>>
    execute_impl(const edm::silicon_cell_collection::const_view& cells,
                 const silicon_detector_description::const_view& det_descr,
                 bool keep_disjoint_set) const;

    /// Clusterization configuration
    config_type m_config;
    /// Memory reserved for edge cases
    mutable vecmem::data::vector_buffer<details::index_t> m_f_backup;
    mutable vecmem::data::vector_buffer<details::index_t> m_gf_backup;
    mutable vecmem::unique_alloc_ptr<unsigned int> m_backup_mutex;
    mutable vecmem::data::vector_buffer<unsigned char> m_adjc_backup;
    mutable vecmem::data::vector_buffer<details::index_t> m_adjv_backup;

};  // class clusterization_algorithm

}  // namespace traccc::device
