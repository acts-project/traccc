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
#include "traccc/clusterization/detail/sparse_ccl.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/utils/algorithm.hpp"

namespace traccc::sycl {

struct component_connection
    : public algorithm<host_cluster_container(const host_cell_container &)> {

    public:
    /// Constructor for component_connection
    ///
    /// @param mr is memory resource
    /// @param queue is the sycl queue for kernel inocation
    component_connection(vecmem::memory_resource &mr, queue_wrapper queue);

    /// Callable operator for component connection for all the cells per event
    ///
    /// @param cells_per_event are the input cells grouped by module
    host_cluster_container operator()(
        const host_cell_container &c) const override;

    /// Callable operator for component connection for all the cells per event
    ///
    /// @param cells_per_event are the input cells grouped by module
    /// @return a container of clusters from all the modules
    void operator()(const host_cell_container &cells_per_event,
                    output_type &clusters) const;

    private:
    std::reference_wrapper<vecmem::memory_resource> m_mr;
    mutable queue_wrapper m_queue;
};

}  // namespace traccc::sycl
