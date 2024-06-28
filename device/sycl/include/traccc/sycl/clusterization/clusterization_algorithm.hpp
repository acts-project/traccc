/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// SYCL library include(s).
#include "traccc/sycl/utils/queue_wrapper.hpp"

// Project include(s).
#include "traccc/edm/cell.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/geometry/detector_description.hpp"
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
          const cell_collection_types::const_view&,
          const detector_description::const_view&)> {

    public:
    /// Constructor for clusterization algorithm
    ///
    /// @param mr is a struct of memory resources (shared or host & device)
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param queue is a wrapper for the for the sycl queue for kernel
    ///              invocation
    /// @param target_cells_per_partition the average number of cells in each
    ///                                   partition
    clusterization_algorithm(const traccc::memory_resource& mr,
                             vecmem::copy& copy, queue_wrapper& queue,
                             const unsigned short target_cells_per_partition);

    /// Callable operator for clusterization algorithm
    ///
    /// @param cells     All cells in an event
    /// @param det_descr The detector description
    /// @return a measurement collection (buffer)
    ///
    output_type operator()(
        const cell_collection_types::const_view& cells,
        const detector_description::const_view& det_descr) const override;

    private:
    /// The average number of cells in each partition
    unsigned short m_target_cells_per_partition;
    /// Memory resource(s) to use in the algorithm
    traccc::memory_resource m_mr;
    /// The copy object to use
    std::reference_wrapper<vecmem::copy> m_copy;
    /// The SYCL queue to use
    std::reference_wrapper<queue_wrapper> m_queue;

};  // class clusterization_algorithm

}  // namespace traccc::sycl
