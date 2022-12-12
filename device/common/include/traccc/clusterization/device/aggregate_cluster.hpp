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
#include "traccc/edm/alt_measurement.hpp"
#include "traccc/edm/ccl_partition.hpp"

// System include(s).
#include <cstddef>

namespace traccc::device {

/// Function which looks for cells which share the same "parent" index and
/// aggregates them into a cluster.
///
/// @param[in] cells    collection of cells
/// @param[in] modules  collection of modules to which the cells are linked to
/// @param[in] f        array of "parent" indices for all cells in this
/// partition
/// @param[in] part     partition this cell belongs to
/// @param[in] tid      current thread id
/// @param[out] out     pointer to position in result clusters array to be
/// filled
/// @param outi         atomic object to store number of added clusters for
/// current thread group
TRACCC_HOST_DEVICE
inline void aggregate_cluster(
    const alt_cell_collection_types::const_device& cells,
    const cell_module_collection_types::const_device& modules,
    const unsigned short* f, const ccl_partition& part,
    const unsigned short tid, alt_measurement* out, unsigned int& outi);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/clusterization/device/impl/aggregate_cluster.ipp"