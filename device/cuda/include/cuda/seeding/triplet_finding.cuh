/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cuda/seeding/detail/doublet_counter.hpp>
#include <cuda/seeding/detail/triplet_counter.hpp>
#include <cuda/utils/definitions.hpp>
#include <edm/internal_spacepoint.hpp>
#include <seeding/detail/doublet.hpp>
#include <seeding/detail/seeding_config.hpp>
#include <seeding/doublet_finding_helper.hpp>
#include <seeding/seed_selecting_helper.hpp>
#include <seeding/triplet_finding_helper.hpp>

namespace traccc {
namespace cuda {

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
void triplet_finding(const seedfinder_config& config,
                     const seedfilter_config& filter_config,
                     host_internal_spacepoint_container& internal_sp_container,
                     host_doublet_counter_container& doublet_counter_container,
                     host_doublet_container& mid_bot_doublet_container,
                     host_doublet_container& mid_top_doublet_container,
                     host_triplet_counter_container& triplet_counter_container,
                     host_triplet_container& triplet_container,
                     vecmem::memory_resource* resource);

}  // namespace cuda
}  // namespace traccc
