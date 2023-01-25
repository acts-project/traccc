/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/alt_cell.hpp"
#include "traccc/edm/partition.hpp"

// System include(s).
#include <cstddef>

namespace traccc::device {

/// Function for looking for adjacent cells. The thread ids will range from 0 to
/// partitioning::MAX_CELLS_PER_PARTITION and the number of blocks will equal
/// the number of partitions, hence checking all cells.
///
/// @param[in] cells    Collection of cells
/// @param[in] tid      Current thread id
/// @param[in] start    Current partition start point
/// @param[in] end      Current partition end point
/// @param[out] ajc     Number of adjacent cells
/// @param[out] ajv     Indices of adjacent cells
///
TRACCC_HOST_DEVICE
inline void reduce_problem_cell(
    const alt_cell_collection_types::const_device& cells,
    const unsigned short tid, const partition start, const partition end,
    unsigned char& adjc, unsigned short adjv[8]);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/clusterization/device/impl/reduce_problem_cell.ipp"