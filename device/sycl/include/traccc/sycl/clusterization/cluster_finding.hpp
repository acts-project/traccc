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
#include "traccc/edm/measurement.hpp"
#include "traccc/utils/algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

namespace traccc::sycl {

class cluster_finding : public algorithm<host_measurement_container(
                            const cell_container_types::host &)> {
    public:
    /// Constructor for cluster_finding
    ///
    /// @param mr is the memory resource
    /// @param queue is the sycl queue for kernel invocation
    cluster_finding(vecmem::memory_resource &mr, queue_wrapper queue);

    /// Callable operator for cluster finding for cells from all the modules
    ///
    /// @param cells_per_event is a container with cell modules as headers
    /// and cells as the items jagged vector
    /// @return a measurement container with cell modules for headers and
    /// measurements as items
    output_type operator()(
        const cell_container_types::host &cells_per_event) const;

    private:
    std::reference_wrapper<vecmem::memory_resource> m_mr;
    mutable queue_wrapper m_queue;
};

}  // namespace traccc::sycl