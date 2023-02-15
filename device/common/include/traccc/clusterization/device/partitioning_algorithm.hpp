/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/alt_cell.hpp"
#include "traccc/edm/device/partition.hpp"
#include "traccc/utils/algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

namespace traccc::device {

/// Partitioning algorithm, dividing cells
///
/// This algorithm divides a collection of cells into similarly sized
/// partitions, to provide an efficient way of distributing work in a
/// GPU-friendly way for the clusterization algorithm
///
class partitioning_algorithm
    : public algorithm<partition_collection_types::host(
          const alt_cell_collection_types::host&)> {

    public:
    /// Partitioning algorithm constructor
    ///
    /// @param mr The memory resource to use for the result objects
    /// @param max_cells_per_partition The number of cells to put together in
    /// each partition
    ///
    partitioning_algorithm(vecmem::memory_resource& mr,
                           const unsigned short max_cells_per_partition);

    /// Construct partitions for the cells
    ///
    /// @param cells A collection of all the cells in a event
    /// @return Similarly sized partitions for the given cells
    ///
    output_type operator()(
        const alt_cell_collection_types::host& cells) const override;

    private:
    /// The number of cells to put together in each partition
    unsigned short m_max_cells_per_partition;
    /// Reference to the host-accessible memory resource
    std::reference_wrapper<vecmem::memory_resource> m_mr;

};  // class clusterization_algorithm

}  // namespace traccc::device
