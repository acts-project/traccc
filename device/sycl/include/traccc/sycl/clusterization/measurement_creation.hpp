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
#include "traccc/clusterization/measurement_creation_helper.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/utils/algorithm.hpp"

namespace traccc::sycl {

struct measurement_creation : public algorithm<host_measurement_container(
                                  const host_cluster_container &,
                                  const host_cell_module_collection &)> {
    public:
    /// Constructor for measurement_creation
    ///
    /// @param mr is the memory resource
    /// @param queue is the sycl queue for kernel invocation
    measurement_creation(vecmem::memory_resource &mr, queue_wrapper queue);

    /// Callable operator for measurement creation for all the modules
    ///
    /// @param clusters are the input cells into the connected component
    /// form all the modules
    /// @param cluster_sizes is a vector of number of clusters per each module
    /// @param cell_modules_per_event is a collection of cell modules
    /// @return a measurement container with cell modules for headers and
    /// measurements as items - usually same size or sometime slightly smaller
    /// than the input (cluster sizes)
    host_measurement_container operator()(
        const host_cluster_container &c,
        const host_cell_module_collection &l) const override;

    /// Callable operator for measurement creation for all the modules
    ///
    /// @param clusters are the input cells into the connected component
    /// form all the modules
    /// @param cluster_sizes is a vector of number of clusters per each module
    /// @param cell_modules_per_event is a collection of cell modules
    /// @return a measurement container with cell modules for headers and
    /// measurements as items - usually same size or sometime slightly smaller
    /// than the input (cluster sizes)
    void operator()(const host_cluster_container &clusters,
                    const host_cell_module_collection &cell_modules_per_event,
                    output_type &measurements) const;

    private:
    std::reference_wrapper<vecmem::memory_resource> m_mr;
    mutable queue_wrapper m_queue;
};

}  // namespace traccc::sycl