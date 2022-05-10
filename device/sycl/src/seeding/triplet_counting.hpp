/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// SYCL library include(s).
#include "traccc/sycl/utils/queue_wrapper.hpp"
#include "triplet_counter.hpp"

// Project include(s).
#include "traccc/edm/device/doublet_counter.hpp"
#include "traccc/seeding/detail/doublet.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

namespace traccc::sycl {

/// Forward declaration of triplet counting function
/// The number of triplets per mid-bot doublets are counted and recorded into
/// triplet counter container
///
/// @param config seed finder config
/// @param mbc_headers The header vector for the middle-bottom doublet counts
/// @param internal_sp_view vecmem container for internal spacepoint
/// @param doublet_counter_container vecmem container for doublet_counter
/// @param mid_bot_doublet_container vecmem container for mid-bot doublets
/// @param mid_top_doublet_container vecmem container for mid-top doublets
/// @param triplet_counter_container vecmem container for triplet counters
/// @param resource vecmem memory resource
/// @param q sycl queue for kernel scheduling
void triplet_counting(const seedfinder_config& config,
                      const vecmem::vector<doublet_per_bin>& mbc_headers,
                      const sp_grid_const_view& internal_sp,
                      const device::doublet_counter_container_types::const_view&
                          doublet_counter_container,
                      doublet_container_view mid_bot_doublet_container,
                      doublet_container_view mid_top_doublet_container,
                      host_triplet_counter_container& triplet_counter_container,
                      vecmem::memory_resource& resource, queue_wrapper queue);

}  // namespace traccc::sycl
