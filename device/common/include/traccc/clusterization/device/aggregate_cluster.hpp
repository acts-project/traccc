/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/alt_measurement.hpp"
#include "traccc/edm/cell.hpp"

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
/// @param[in] start    partition start point this cell belongs to
/// @param[in] end      partition end point this cell belongs to
/// @param[in] cid      current cell id
/// @param[out] out     cluster to fill
TRACCC_HOST_DEVICE
inline void aggregate_cluster(
    const cell_collection_types::const_device& cells,
    const cell_module_collection_types::const_device& modules,
    const vecmem::data::vector_view<unsigned short> f_view,
    const unsigned int start, const unsigned int end, const unsigned short cid,
    alt_measurement& out, vecmem::data::vector_view<unsigned int> cell_links,
    const unsigned int link);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/clusterization/device/impl/aggregate_cluster.ipp"