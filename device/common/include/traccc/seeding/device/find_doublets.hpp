/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/fill_prefix_sum.hpp"
#include "traccc/edm/device/doublet_counter_spM.hpp"
#include "traccc/seeding/detail/doublet_spM.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"

// System include(s).
#include <cstddef>

namespace traccc::device {

/// Function finding all of the spacepoint doublets
///
/// Based on the information collected by @c traccc::device::count_doublets it
/// can fill containers with the specific doublet pairs that exist in the event.
///
/// @param[in] globalIndex       The index of the current thread
/// @param[in] config            Seedfinder configuration
/// @param[in] sp_view           The spacepoint grid to count doublets on
/// @param[in] dc_view           Collection with the number of doublets to find
/// @param[out] mb_doublets_view Container of middle-bottom doublets
/// @param[out] mt_doublets_view Container of middle-top doublets
///
TRACCC_HOST_DEVICE
inline void find_doublets(
    const std::size_t globalIndex, const seedfinder_config& config,
    const sp_grid_const_view& sp_view,
    const device::doublet_counter_spM_collection_types::const_view& dc_view,
    doublet_spM_container_types::view mb_doublets_view,
    doublet_spM_container_types::view mt_doublets_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/seeding/device/impl/find_doublets.ipp"
