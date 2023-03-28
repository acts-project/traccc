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

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <functional>

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
class component_connection : public algorithm<cluster_container_types::host(
                                 const cell_collection_types::host&)> {

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
    /// @param cells Collection of input cells sorted by module
    ///
    /// c++20 piping interface:
    /// @return a cluster collection
    ///
    output_type operator()(
        const cell_collection_types::host& cells) const override;

    /// @}

    private:
    /// The memory resource used by the algorithm
    std::reference_wrapper<vecmem::memory_resource> m_mr;

};  // class component_connection

}  // namespace traccc
