/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/alt_cell.hpp"
#include "traccc/edm/ccl_partition.hpp"
#include "traccc/utils/algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

namespace traccc {

/// Partitioning algorithm, dividing cells
///
/// This algorithm divides a collection of cells into similarly sized
/// partitions, to provide an efficient way of distributing work in a
/// GPU-friendly way for the clusterization algorithm
///
class partitioning_algorithm
    : public algorithm<ccl_partition_collection_types::host(
          const alt_cell_collection_types::host&,
          const cell_module_collection_types::host&)> {

    public:
    /// Partitioning algorithm constructor
    ///
    /// @param mr The memory resource to use for the result objects
    ///
    partitioning_algorithm(vecmem::memory_resource& mr);

    /// Construct partitions for the cells
    ///
    /// @param cells A collection of all the cells in a event
    /// @param modules The modules the cells belong to
    /// @return Similarly sized partitions for the given cells
    ///
    output_type operator()(
        const alt_cell_collection_types::host& cells,
        const cell_module_collection_types::host& modules) const override;

    private:
    /// Reference to the host-accessible memory resource
    std::reference_wrapper<vecmem::memory_resource> m_mr;

};  // class clusterization_algorithm

}  // namespace traccc
