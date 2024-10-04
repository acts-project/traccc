/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/edm/silicon_cluster_collection.hpp"
#include "traccc/utils/algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <functional>

namespace traccc::host {

/// Pixel cell clusterization based on a SparseCCL algorithm
///
/// The implementation is based on the paper:
/// https://doi.org/10.1109/DASIP48288.2019.9049184
///
class sparse_ccl_algorithm
    : public algorithm<edm::silicon_cluster_collection::host(
          const edm::silicon_cell_collection::const_view&)> {

    public:
    /// Constructor for component_connection
    ///
    /// @param mr is the memory resource
    ///
    sparse_ccl_algorithm(vecmem::memory_resource& mr);

    /// @name Operator(s) to use in host code
    /// @{

    /// Callable operator for the connected component labelling
    ///
    /// @param cells_view Collection of input cells sorted by module
    ///
    /// @return a cluster container
    ///
    output_type operator()(const edm::silicon_cell_collection::const_view&
                               cells_view) const override;

    /// @}

    private:
    /// The memory resource used by the algorithm
    std::reference_wrapper<vecmem::memory_resource> m_mr;

};  // class sparse_ccl_algorithm

}  // namespace traccc::host
