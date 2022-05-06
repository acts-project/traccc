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
        const cell_container_types::host& cells_per_event) const override {

        output_type measurements_per_event(&m_mr.get());

        // 2 step cluster finding algorithm - returns the measurments
        measurements_per_event = cf->operator()(cells_per_event);

        return measurements_per_event;
    }

    private:
    std::reference_wrapper<vecmem::memory_resource> m_mr;
    mutable queue_wrapper m_queue;
};

}  // namespace traccc::sycl