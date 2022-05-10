/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/utils/algorithm.hpp"

namespace traccc {

/// Connected component labelling
///
/// Note that the separation between the public and private interface is
/// only there in the class because the compilers can't automatically figure
/// out the "vector type" of the templated implementation, without adding a
/// lot of "internal knowledge" about the vector types into this piece of
/// code. So instead the public operators are specifically implemented for
/// the host- and device versions of the EDM, making use of a single
/// implementation internally.
///
class component_connection
    : public algorithm<host_cluster_container(
          const cell_collection_types::host&, const cell_module&)> {

    public:
    /// Constructor for component_connection
    ///
    /// @param mr is the memory resource
    component_connection(vecmem::memory_resource& mr) : m_mr(mr) {}

    /// @name Operator(s) to use in host code
    /// @{

    /// Callable operator for the connected component, based on one single
    /// module
    ///
    /// @param cells are the input cells into the connected component, they are
    ///              per module and unordered
    /// @param module The description of the module that the cells belong to
    ///
    /// c++20 piping interface:
    /// @return a cluster collection
    ///
    host_cluster_container operator()(const cell_collection_types::host& cells,
                                      const cell_module& module) const override;

    /// @}

    private:
    std::reference_wrapper<vecmem::memory_resource> m_mr;

};  // class component_connection

}  // namespace traccc
