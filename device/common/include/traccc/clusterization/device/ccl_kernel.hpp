/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/alt_measurement.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"

// Vecmem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <cstddef>

namespace traccc::device {

namespace {
/// These indices in clusterization will only range from 0 to
/// max_cells_per_partition, so we only need a short
using index_t = unsigned short;

static constexpr int TARGET_CELLS_PER_THREAD = 8;
static constexpr int MAX_CELLS_PER_THREAD = 12;
}  // namespace

/// Function which reads raw detector cells and turns them into measurements.
///
/// @param[in] threadId current thread index
/// @param[in] blckDim  current thread block size
/// @param[in] blckId   current thread block index
/// @param[in] cells_view    collection of cells
/// @param[in] modules_view  collection of modules to which the cells are linked
/// @param[in] max_cells_per_partition maximum number of cells per thread block
/// @param[in] target_cells_per_partition average number of cells per thread
/// block
/// @param partition_start    partition start point for this thread block
/// @param partition_end      partition end point for this thread block
/// @param outi               number of measurements for this partition
/// @param f  array of "parent" indices for all cells in this partition
/// @param gf array of "grandparent" indices for all cells in this partition
/// @param barrier  A generic object for block-wide synchronisation
/// @param[out] measurements_view collection of measurements
/// @param[out] measurement_count number of measurements
/// @param[out] cell_links    collection of links to measurements each cell is
/// put into
template <typename barrier_t>
TRACCC_DEVICE inline void ccl_kernel(
    const index_t threadId, const index_t blckDim, const unsigned int blockId,
    const cell_collection_types::const_view cells_view,
    const cell_module_collection_types::const_view modules_view,
    const index_t max_cells_per_partition,
    const index_t target_cells_per_partition, unsigned int& partition_start,
    unsigned int& partition_end, unsigned int& outi, index_t* f, index_t* gf,
    barrier_t& barrier,
    alt_measurement_collection_types::view measurements_view,
    unsigned int& measurement_count,
    vecmem::data::vector_view<unsigned int> cell_links);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/clusterization/device/impl/ccl_kernel.ipp"