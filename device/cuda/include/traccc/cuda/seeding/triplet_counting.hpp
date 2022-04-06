/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/cuda/seeding/detail/doublet_counter.hpp"
#include "traccc/cuda/seeding/detail/triplet_counter.hpp"
#include "traccc/edm/internal_spacepoint.hpp"
#include "traccc/seeding/detail/doublet.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/seeding/detail/triplet.hpp"
#include "traccc/seeding/doublet_finding_helper.hpp"
#include "traccc/seeding/triplet_finding_helper.hpp"

namespace traccc {
namespace cuda {

/// Forward declaration of triplet counting function
/// The number of triplets per mid-bot doublets are counted and recorded into
/// triplet counter container
///
/// @param config seed finder config
/// @param internal_sp_view vecmem container for internal spacepoint
/// @param doublet_counter_container vecmem container for doublet_counter
/// @param mid_bot_doublet_container vecmem container for mid-bot doublets
/// @param mid_top_doublet_container vecmem container for mid-top doublets
/// @param triplet_counter_container vecmem container for triplet counters
/// @resource vecmem memory resource
void triplet_counting(const seedfinder_config& config,
                      const vecmem::vector<doublet_per_bin>& mbc_headers,
                      sp_grid_view internal_sp_view,
                      doublet_counter_container_view dcc_view,
                      doublet_container_view mbc_view,
                      doublet_container_view mtc_view,
                      triplet_counter_container_view tcc_view,
                      vecmem::memory_resource& resource);

}  // namespace cuda
}  // namespace traccc
