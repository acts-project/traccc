/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// clusterization
#include "traccc/sycl/clusterization/measurement_creation.hpp"

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

        mt = std::make_shared<traccc::sycl::measurement_creation>(
            traccc::sycl::measurement_creation(mr, q));
    }

    output_type operator()(
        const host_cell_container& cells_per_event) const override {

        output_type measurements_per_event(&m_mr.get());

        // Measurement creation
        measurements_per_event = mt->operator()(cells_per_event);

        return measurements_per_event;
    }

    private:
    // algorithms
    std::shared_ptr<traccc::sycl::measurement_creation> mt;
    std::reference_wrapper<vecmem::memory_resource> m_mr;
};

}  // namespace traccc::sycl