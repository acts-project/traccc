/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/clusterization/measurement_creation_algorithm.hpp"
#include "traccc/clusterization/sparse_ccl_algorithm.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/utils/algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <functional>
#include <variant>

namespace traccc::host {

/// Clusterization algorithm, creating measurements from cells
///
/// This algorithm creates local/2D measurements separately for each detector
/// module from the cells of the modules.
///
class clusterization_algorithm
    : public algorithm<measurement_collection_types::host(
          const cell_collection_types::const_view&,
          const cell_module_collection_types::const_view&)> {

    public:
    using config_type = std::monostate;

    /// Clusterization algorithm constructor
    ///
    /// @param mr The memory resource to use for the result objects
    ///
    clusterization_algorithm(vecmem::memory_resource& mr);

    /// Construct measurements for each detector module
    ///
    /// @param cells_view The cells for every detector module in the event
    /// @param modules_view A collection of detector modules
    /// @return The measurements reconstructed for every detector module
    ///
    output_type operator()(const cell_collection_types::const_view& cells_view,
                           const cell_module_collection_types::const_view&
                               modules_view) const override;

    private:
    /// @name Sub-algorithms used by this algorithm
    /// @{

    /// Per-module cluster creation algorithm
    sparse_ccl_algorithm m_cc;

    /// Per-module measurement creation algorithm
    measurement_creation_algorithm m_mc;

    /// @}

    /// Reference to the host-accessible memory resource
    std::reference_wrapper<vecmem::memory_resource> m_mr;

};  // class clusterization_algorithm

}  // namespace traccc::host
