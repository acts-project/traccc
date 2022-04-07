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

/// Forward declaration of doublet counting function
/// The number of mid-bot and mid-top doublets are counted for all spacepoints
/// and recorded into doublet counter container if the number of doublets are
/// larger than zero.
///
/// @param config seed finder config
/// @param internal_sp_container vecmem container for internal spacepoint
/// @param doublet_counter_container vecmem container for doublet_counter
/// @param resource vecmem memory resource
void doublet_counting(const seedfinder_config& config,
                      sp_grid_view internal_sp_view,
                      doublet_counter_container_view dcc_view,
                      vecmem::memory_resource& resource);

}  // namespace cuda
}  // namespace traccc
