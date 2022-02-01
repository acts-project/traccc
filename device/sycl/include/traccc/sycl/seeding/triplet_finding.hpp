/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// SYCL include(s).
#include <CL/sycl.hpp>

// Project include(s).
#include "traccc/edm/internal_spacepoint.hpp"
#include "traccc/seeding/detail/doublet.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/seeding/doublet_finding_helper.hpp"
#include "traccc/seeding/seed_selecting_helper.hpp"
#include "traccc/seeding/triplet_finding_helper.hpp"
#include "traccc/sycl/seeding/detail/doublet_counter.hpp"
#include "traccc/sycl/seeding/detail/sycl_helper.hpp"
#include "traccc/sycl/seeding/detail/triplet_counter.hpp"

namespace traccc {
namespace sycl {

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
/// @param q sycl queue for kernel scheduling
void triplet_finding(const seedfinder_config& config,
                     const seedfilter_config& filter_config,
                     sp_grid& internal_sp,
                     host_doublet_counter_container& doublet_counter_container,
                     host_doublet_container& mid_bot_doublet_container,
                     host_doublet_container& mid_top_doublet_container,
                     host_triplet_counter_container& triplet_counter_container,
                     host_triplet_container& triplet_container,
                     vecmem::memory_resource& resource, ::sycl::queue* q);

}  // namespace sycl
}  // namespace traccc
