/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// SYCL library include(s).
#include "traccc/sycl/utils/queue_wrapper.hpp"

#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/utils/algorithm.hpp"

namespace traccc::sycl {

/// Connected component labeling.
struct measurement_creation
    : public algorithm<host_measurement_container(
          const host_cluster_container &, const std::vector<std::size_t> &,
          const host_cell_container &)> {
    public:
    /// Constructor for measurement_creation
    ///
    /// @param mr is the memory resource
    measurement_creation(vecmem::memory_resource &mr, queue_wrapper queue); 

    /// Callable operator for the connected component, based on one single
    /// module
    ///
    /// @param clusters are the input cells into the connected component, they
    /// are
    ///              per module and unordered
    ///
    /// C++20 piping interface
    ///
    /// @return a measurement collection - usually same size or sometime
    /// slightly smaller than the input
    host_measurement_container operator()(
        const host_cluster_container &c,
        const std::vector<std::size_t> &s,
        const host_cell_container &l) const override; 

    /// Callable operator for the connected component, based on one single
    /// module
    ///
    /// @param clusters are the input cells into the connected component, they
    /// are
    ///              per module and unordered
    ///
    /// void interface
    ///
    /// @return a measurement collection - usually same size or sometime
    /// slightly smaller than the input
    void operator()(const host_cluster_container &clusters,
                    const std::vector<std::size_t> &cluster_sizes,
                    const host_cell_container &cells_per_event,
                    output_type &measurements) const; 

    private:
    std::reference_wrapper<vecmem::memory_resource> m_mr;
    mutable queue_wrapper m_queue;
};

}  // namespace traccc::sycl