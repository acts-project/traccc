/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/hints.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/barrier.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"

// Vecmem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <cstddef>

namespace traccc::device {

namespace details {

/// These indices in clusterization will only range from 0 to
/// max_cells_per_partition, so we only need a short
using index_t = unsigned short;

static constexpr int TARGET_CELLS_PER_THREAD = 8;
static constexpr int MAX_CELLS_PER_THREAD = 32;

/// Helper struct for calculating some of the input parameters of @c ccl_kernel
struct ccl_kernel_helper {

    /// Constructor setting the helper parameters
    ///
    /// @param[in] target_cells_per_partition Target average number of cells per
    ///                                       thread block
    /// @param[in] n_cells Total number of cells
    ///
    ccl_kernel_helper(index_t target_cells_per_partition,
                      unsigned int n_cells) {

        max_cells_per_partition =
            (target_cells_per_partition * MAX_CELLS_PER_THREAD +
             TARGET_CELLS_PER_THREAD - 1) /
            TARGET_CELLS_PER_THREAD;
        threads_per_partition =
            (target_cells_per_partition + TARGET_CELLS_PER_THREAD - 1) /
            TARGET_CELLS_PER_THREAD;
        num_partitions = (n_cells + target_cells_per_partition - 1) /
                         target_cells_per_partition;
    }

    /// Maximum number of cells per partition
    index_t max_cells_per_partition;
    /// Number of threads per partition
    unsigned int threads_per_partition;
    /// Number of partitions
    unsigned int num_partitions;

};  // struct ccl_kernel_helper

}  // namespace details

/// Function which reads raw detector cells and turns them into measurements.
///
/// @param[in] threadId current thread index
/// @param[in] blckDim  current thread block size
/// @param[in] blckId   current thread block index
/// @param[in] cells_view    collection of cells
/// @param[in] modules_view  collection of modules to which the cells are linked
/// @param[in] max_cells_per_partition maximum number of cells per thread block
/// @param[in] target_cells_per_partition average number of cells per thread
///                                       block
/// @param partition_start    partition start point for this thread block
/// @param partition_end      partition end point for this thread block
/// @param outi               number of measurements for this partition
/// @param f_view  array of "parent" indices for all cells in this partition
/// @param gf_view array of "grandparent" indices for all cells in this
///                partition
/// @param barrier  A generic object for block-wide synchronisation
/// @param[out] measurements_view collection of measurements
/// @param[out] cell_links    collection of links to measurements each cell is
/// put into
template <TRACCC_CONSTRAINT(device::concepts::barrier) barrier_t>
TRACCC_DEVICE inline void ccl_kernel(
    details::index_t threadId, details::index_t blckDim, unsigned int blockId,
    const cell_collection_types::const_view cells_view,
    const cell_module_collection_types::const_view modules_view,
    const details::index_t max_cells_per_partition,
    const details::index_t target_cells_per_partition,
    unsigned int& partition_start, unsigned int& partition_end,
    unsigned int& outi, vecmem::data::vector_view<details::index_t> f_view,
    vecmem::data::vector_view<details::index_t> gf_view, barrier_t& barrier,
    measurement_collection_types::view measurements_view,
    vecmem::data::vector_view<unsigned int> cell_links);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/clusterization/device/impl/ccl_kernel.ipp"
