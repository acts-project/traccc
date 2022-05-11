/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// SYCL library include(s).
#include "traccc/sycl/utils/queue_wrapper.hpp"

// Project include(s).
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/utils/algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

namespace traccc::sycl {

class clusterization_algorithm : public algorithm<host_measurement_container(
                                     const cell_container_types::host&)> {

    public:
    /// Constructor for clusterization algorithm
    ///
    /// @param mr is the memory resource
    /// @param queue is the sycl queue for kernel invocation
    clusterization_algorithm(vecmem::memory_resource& mr, queue_wrapper queue);

    /// Callable operator for clusterization algorithm
    ///
    /// @param cells_per_event is a container with cell modules as headers
    /// and cells as the items
    /// @return a measurement container with cell modules as headers and
    /// measurements as items
    output_type operator()(
        const cell_container_types::host& cells_per_event) const override;

    private:
    std::reference_wrapper<vecmem::memory_resource> m_mr;
    mutable queue_wrapper m_queue;
};

}  // namespace traccc::sycl