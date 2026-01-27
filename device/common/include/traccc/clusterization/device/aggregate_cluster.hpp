/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/hints.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

// System include(s).
#include <optional>

namespace traccc::device {

/// Function which looks for cells which share the same "parent" index and
/// aggregates them into a cluster.
///
/// @param[in] cells     collection of cells
/// @param[in] det_descr The detector description
/// @param[in] f         array of "parent" indices for all cells in this
///                      partition
/// @param[in] start     partition start point this cell belongs to
/// @param[in] end       partition end point this cell belongs to
/// @param[in] cid       current cell id
/// @param[out] out      cluster to fill
/// @param[out] disjoint_set Array of unsigned integers of
///                      length $|cells|$ to which an integer is written
///                      identifying the measurement index to which each cell
///                      belongs.
/// @param[out] cluster_size Optional integer which is filled with the size of
///                      the measurement that is created.
///
TRACCC_HOST_DEVICE inline void aggregate_cluster(
    const clustering_config& cfg,
    const edm::silicon_cell_collection::const_device& cells,
    const silicon_detector_description::const_device& det_descr,
    const vecmem::device_vector<details::index_t>& f, unsigned int start,
    unsigned int end, unsigned short cid,
    edm::measurement_collection<default_algebra>::device::proxy_type out,
    vecmem::data::vector_view<unsigned int> cell_links, unsigned int link,
    vecmem::device_vector<unsigned int>& disjoint_set,
    std::optional<std::reference_wrapper<unsigned int>> cluster_size);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/clusterization/device/impl/aggregate_cluster.ipp"
