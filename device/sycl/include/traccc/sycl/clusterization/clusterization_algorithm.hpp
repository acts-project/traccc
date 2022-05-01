/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// clusterization
#include "traccc/sycl/clusterization/cluster_finding.hpp"

namespace traccc::sycl {

class clusterization_algorithm
    : public algorithm<host_measurement_container(const host_cell_container&)> {

    public:
    /// Constructor for clusterization algorithm
    ///
    /// @param mr is the memory resource
    clusterization_algorithm(vecmem::memory_resource& mr,
                             ::sycl::queue* q = nullptr)
        : m_mr(mr) {

        cf = std::make_shared<traccc::sycl::cluster_finding>(
            traccc::sycl::cluster_finding(mr, q));
    }

    output_type operator()(
        const host_cell_container& cells_per_event) const override {

        output_type measurements_per_event(&m_mr.get());

        // Cluster finding algorithm - returns the measurments
        measurements_per_event = cf->operator()(cells_per_event);

        return measurements_per_event;
    }

    private:
    // algorithms
    std::shared_ptr<traccc::sycl::cluster_finding> cf;
    std::reference_wrapper<vecmem::memory_resource> m_mr;
};

}  // namespace traccc::sycl