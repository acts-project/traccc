/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// SYCL library include(s).
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/clusterization/device/ccl_kernel_definitions.hpp"
#include "traccc/sycl/utils/queue_wrapper.hpp"

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <functional>

namespace traccc::sycl {

/// Algorithm performing hit clusterization
///
/// This algorithm implements hit clusterization in a massively-parallel
/// approach. Each thread handles a pre-determined number of detector cells.
///
/// This algorithm returns a buffer which is not necessarily filled yet. A
/// synchronisation statement is required before destroying the buffer.
///
class clusterization_algorithm
    : public algorithm<measurement_collection_types::buffer(
          const edm::silicon_cell_collection::const_view&,
          const silicon_detector_description::const_view&)> {

    public:
    /// Configuration type
    using config_type = clustering_config;

    /// Constructor for clusterization algorithm
    ///
    /// @param mr is a struct of memory resources (shared or host & device)
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param queue is a wrapper for the for the sycl queue for kernel
    ///              invocation
    /// @param config the clustering configuration
    clusterization_algorithm(const traccc::memory_resource& mr,
                             vecmem::copy& copy, queue_wrapper& queue,
                             const config_type& config);

    /// Callable operator for clusterization algorithm
    ///
    /// @param cells     All cells in an event
    /// @param det_descr The detector description
    /// @return a measurement collection (buffer)
    ///
    output_type operator()(
        const edm::silicon_cell_collection::const_view& cells,
        const silicon_detector_description::const_view& det_descr)
        const override;

    private:
    /// Memory resource(s) to use in the algorithm
    traccc::memory_resource m_mr;
    /// The SYCL queue to use
    std::reference_wrapper<queue_wrapper> m_queue;
    /// The copy object to use
    std::reference_wrapper<vecmem::copy> m_copy;
    /// The average number of cells in each partition
    const config_type m_config;
    /// Memory reserved for edge cases
    vecmem::data::vector_buffer<device::details::index_t> m_f_backup,
        m_gf_backup;
    vecmem::data::vector_buffer<unsigned char> m_adjc_backup;
    vecmem::data::vector_buffer<device::details::index_t> m_adjv_backup;
    vecmem::unique_alloc_ptr<unsigned int> m_backup_mutex;

};  // class clusterization_algorithm
}  // namespace traccc::sycl
