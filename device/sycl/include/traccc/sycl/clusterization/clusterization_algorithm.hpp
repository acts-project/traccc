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
#include "traccc/edm/cell.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <memory>

namespace traccc::sycl {

class clusterization_algorithm
    : public algorithm<measurement_collection_types::buffer(
          const cell_collection_types::const_view&,
          const cell_module_collection_types::const_view&)> {

    public:
    /// Configuration type
    using config_type = clustering_config;

    /// Constructor for clusterization algorithm
    ///
    /// @param mr is a struct of memory resources (shared or host & device)
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param queue is a wrapper for the for the sycl queue for kernel
    /// invocation
    /// @param config the clustering configuration
    clusterization_algorithm(const traccc::memory_resource& mr,
                             vecmem::copy& copy, queue_wrapper queue,
                             const config_type& config);

    /// @param cells        a collection of cells
    /// @param modules      a collection of modules
    /// @return a spacepoint collection (buffer) and a collection (buffer) of
    /// links from cells to the spacepoints they belong to.
    output_type operator()(
        const cell_collection_types::const_view& cells,
        const cell_module_collection_types::const_view& modules) const override;

    private:
    /// Memory resource(s) to use in the algorithm
    traccc::memory_resource m_mr;
    /// The SYCL queue to use
    mutable queue_wrapper m_queue;
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
