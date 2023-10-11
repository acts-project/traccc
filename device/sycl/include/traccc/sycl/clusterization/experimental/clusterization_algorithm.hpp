/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// SYCL library include(s).
#include "traccc/sycl/utils/queue_wrapper.hpp"

// Project include(s).
#include "traccc/edm/cell.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

namespace traccc::sycl::experimental {

/// Algorithm performing hit clusterization
///
/// This algorithm implements hit clusterization in a massively-parallel
/// approach. Each thread handles a pre-determined number of detector cells.
///
/// This algorithm returns a buffer which is not necessarily filled yet. A
/// synchronisation statement is required before destroying this buffer.
///
class clusterization_algorithm
    : public algorithm<measurement_collection_types::buffer(
          const cell_collection_types::const_view&,
          const cell_module_collection_types::const_view&)> {

    public:
    /// Constructor for clusterization algorithm
    ///
    /// @param mr The memory resource(s) to use in the algorithm
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param str The CUDA stream to perform the operations in
    /// @param target_cells_per_partition the average number of cells in each
    /// partition
    ///
    clusterization_algorithm(const traccc::memory_resource& mr,
                             vecmem::copy& copy, queue_wrapper queue,
                             const unsigned short target_cells_per_partition);
    // const unsigned short target_cells_per_partition);

    /// Callable operator for clusterization algorithm
    ///
    /// @param cells        a collection of cells
    /// @param modules      a collection of modules
    /// @return a spacepoint collection (buffer) and a collection (buffer)
    /// of links from cells to the spacepoints they belong to.
    output_type operator()(
        const cell_collection_types::const_view& cells,
        const cell_module_collection_types::const_view& modules) const override;

    private:
    /// The average number of cells in each partition
    unsigned short m_target_cells_per_partition;
    /// The maximum number of threads in a work group
    unsigned int m_max_work_group_size;

    /// The memory resource(s) to use
    traccc::memory_resource m_mr;
    /// The copy object to use
    vecmem::copy& m_copy;
    /// The SYCL queue object
    mutable queue_wrapper m_queue;
};

}  // namespace traccc::sycl::experimental