/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/kokkos/seeding/detail/triplet_counter.hpp"
#include "traccc/edm/device/doublet_counter.hpp"
#include "traccc/edm/internal_spacepoint.hpp"
#include "traccc/seeding/detail/doublet.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/seeding/doublet_finding_helper.hpp"
#include "traccc/seeding/seed_selecting_helper.hpp"
#include "traccc/seeding/triplet_finding_helper.hpp"

namespace traccc {
namespace kokkos {

/// Forward declaration of triplet finding function
/// The triplets per mid-bot doublets are found for the compatible mid-bot
/// doublets which were recorded during triplet_counting
///
/// @param config seed finder config
/// @param filter_config seed filter config
/// @param internal_sp_view vecmem container for internal spacepoint
/// @param doublet_counter_container vecmem container for doublet_counter
/// @param mid_bot_doublet_container vecmem container for mid-bot doublets
/// @param mid_top_doublet_container vecmem container for mid-top doublets
/// @param triplet_counter_container vecmem container for triplet counters
/// @param triplet_container vecmem container for triplets
/// @param resource vecmem memory resource
void triplet_finding(
    const seedfinder_config& config, const seedfilter_config& filter_config,
    const vecmem::vector<triplet_counter_per_bin>& tcc_headers,
    sp_grid_const_view internal_sp_view,
    device::doublet_counter_container_types::const_view dcc_view,
    doublet_container_view mbc_view, doublet_container_view mtc_view,
    triplet_counter_container_view tcc_view, triplet_container_view tc_view,
    vecmem::memory_resource& resource);

}  // namespace kokkos
}  // namespace traccc
