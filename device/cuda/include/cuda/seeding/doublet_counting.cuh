/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cuda/seeding/detail/doublet_counter.hpp>
#include <edm/internal_spacepoint.hpp>
#include <seeding/detail/doublet.hpp>
#include <seeding/detail/seeding_config.hpp>
#include <seeding/doublet_finding_helper.hpp>

namespace traccc {
namespace cuda {

/// Forward declaration of doublet counting function
///The number of mid-bot and mid-top doublets are counted for all spacepoints and recorded into doublet counter container if the number of doublets are larger than zero. 
///    
/// @param config seed finder config
/// @param internal_sp_container vecmem container for internal spacepoint
/// @param doublet_counter_container vecmem container for doublet_counter
/// @param resource vecmem memory resource    
void doublet_counting(const seedfinder_config& config,
                      host_internal_spacepoint_container& internal_sp_container,
                      host_doublet_counter_container& doublet_counter_container,
                      vecmem::memory_resource* resource);

}  // namespace cuda
}  // namespace traccc
