/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/hip/utils/stream.hpp"

// Project include(s).
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/clusterization/device/ccl_kernel_definitions.hpp"
#include "traccc/clusterization/device/tags.hpp"
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

namespace traccc::hip {

/// Algorithm performing hit clusterization
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
          device::clustering_discard_disjoint_set&&)>,
      public algorithm<
          std::pair<edm::measurement_collection<default_algebra>::buffer,
                    edm::silicon_cluster_collection::buffer>(
              const edm::silicon_cell_collection::const_view&,
              const silicon_detector_description::const_view&,
              device::clustering_keep_disjoint_set&&)>,
      public messaging {

    public:
    /// Configuration type
    using config_type = clustering_config;

    /// Constructor for clusterization algorithm
    ///
    /// @param mr The memory resource(s) to use in the algorithm
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param str The HIP stream to perform the operations in
    /// @param config The clustering configuration
    /// partition
    ///
    clusterization_algorithm(
        const traccc::memory_resource& mr, vecmem::copy& copy, stream& str,
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
        device::clustering_discard_disjoint_set&&) const override;

    std::pair<edm::measurement_collection<default_algebra>::buffer,
              edm::silicon_cluster_collection::buffer>
    operator()(const edm::silicon_cell_collection::const_view& cells,
               const silicon_detector_description::const_view& det_descr,
               device::clustering_keep_disjoint_set&&) const override;
    /// @}

    private:
    std::pair<edm::measurement_collection<default_algebra>::buffer,
              std::optional<edm::silicon_cluster_collection::buffer>>
    execute_impl(const edm::silicon_cell_collection::const_view& cells,
                 const silicon_detector_description::const_view& det_descr,
                 bool keep_disjoint_set) const;

    /// The memory resource(s) to use
    traccc::memory_resource m_mr;
    /// The copy object to use
    std::reference_wrapper<vecmem::copy> m_copy;
    /// The HIP stream to use
    std::reference_wrapper<stream> m_stream;
    /// The average number of cells in each partition
    const config_type m_config;
    /// Memory reserved for edge cases
    vecmem::data::vector_buffer<device::details::index_t> m_f_backup,
        m_gf_backup;
    vecmem::unique_alloc_ptr<unsigned int> m_backup_mutex;
    vecmem::data::vector_buffer<unsigned char> m_adjc_backup;
    vecmem::data::vector_buffer<device::details::index_t> m_adjv_backup;
};

}  // namespace traccc::hip
