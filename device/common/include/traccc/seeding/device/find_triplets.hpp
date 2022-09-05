/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/get_prefix_sum.hpp"
#include "traccc/edm/device/triplet_counter.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/seeding/detail/triplet.hpp"

// System include(s).
#include <cstddef>

namespace traccc::device {

/// Function finding all of the spacepoint triplets
///
/// Based on the information collected by @c traccc::device::count_triplets it
/// can fill container with the specific triplets exist in the event.
///
/// @param[in] globalIndex       The index of the current thread
/// @param[in] config            Seedfinder configuration
/// @param[in] sp_view           The spacepoint grid to count triplets on
/// @param[in] dc_view           Container with the number of doublets per bin
/// @param[in] mid_bot_doublet_view Container with the mid bottom doublets
/// @param[in] mid_top_doublet_view Container with the mid top doublets
/// @param[in] tc_view           Container with the number of triplets to find
/// @param[in] triplet_ps_view   Prefix sum for @c triplet_view
/// @param[out] triplet_view     Container of triplets
///
TRACCC_HOST_DEVICE
void find_triplets(
    const std::size_t globalIndex, const seedfinder_config& config,
    const seedfilter_config& filter_config, const sp_grid_const_view& sp_view,
    const device::doublet_counter_container_types::const_view& dc_view,
    const doublet_container_view& mid_bot_doublet_view,
    const doublet_container_view& mid_top_doublet_view,
    const device::triplet_counter_container_types::const_view& tc_view,
    const vecmem::data::vector_view<const prefix_sum_element_t>&
        triplet_ps_view,
    triplet_container_view triplet_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/seeding/device/impl/find_triplets.ipp"
