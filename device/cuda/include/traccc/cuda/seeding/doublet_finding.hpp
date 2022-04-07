/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/cuda/seeding/detail/doublet_counter.hpp"
#include "traccc/edm/internal_spacepoint.hpp"
#include "traccc/seeding/common/doublet.hpp"
#include "traccc/seeding/common/doublet_finding_helper.hpp"
#include "traccc/seeding/common/seeding_config.hpp"
#include "traccc/seeding/common/spacepoint_grid.hpp"

namespace traccc {
namespace cuda {

/// Forward declaration of doublet finding function
/// The mid-bot and mid-top doublets are found for the compatible middle
/// spacepoints which was recorded by doublet_counting
///
/// @param config seed finder config
/// @param internal_sp_container vecmem container for internal spacepoint
/// @param doublet_counter_container vecmem container for doublet_counter
/// @param mid_bot_doublet_container vecmem container for mid-bot doublets
/// @param mid_top_doublet_container vecmem container for mid-top doublets
/// @param resource vecmem memory resource
void doublet_finding(const seedfinder_config& config,
                     const vecmem::vector<doublet_counter_per_bin>& dcc_headers,
                     sp_grid_view internal_sp_view,
                     doublet_counter_container_view dcc_view,
                     doublet_container_view mbc_view,
                     doublet_container_view mtc_view,
                     vecmem::memory_resource& resource);

}  // namespace cuda
}  // namespace traccc
