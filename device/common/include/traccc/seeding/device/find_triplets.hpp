/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/fill_prefix_sum.hpp"
#include "traccc/edm/device/device_triplet.hpp"
#include "traccc/edm/device/doublet_counter.hpp"
#include "traccc/edm/device/triplet_counter.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"

// System include(s).
#include <cstddef>

namespace traccc::device {

/// Function finding all of the spacepoint triplets
///
/// Based on the information collected by @c traccc::device::count_triplets it
/// can fill collection with the specific triplets that exist in the event.
///
/// @param[in] globalIndex       The index of the current thread
/// @param[in] config            Seedfinder configuration
/// @param[in] filter_config     Seedfilter configuration
/// @param[in] sp_view           The spacepoint grid to find triplets on
/// @param[in] dc_view           Collection of doublet counters
/// @param[in] mid_top_doublet_view Collection with the mid top doublets
/// @param[in] spM_tc_view       Collection with the number of triplets per spM
/// @param[in] tc_view           Collection with the number of triplets per
/// midBot doublet
/// @param[out] triplet_view     Collection of triplets
///
TRACCC_HOST_DEVICE
inline void find_triplets(
    std::size_t globalIndex, const seedfinder_config& config,
    const seedfilter_config& filter_config, const sp_grid_const_view& sp_view,
    const doublet_counter_collection_types::const_view& dc_view,
    const device_doublet_collection_types::const_view& mid_top_doublet_view,
    const triplet_counter_spM_collection_types::const_view& spM_tc_view,
    const triplet_counter_collection_types::const_view& tc_view,
    device_triplet_collection_types::view triplet_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/seeding/device/impl/find_triplets.ipp"
