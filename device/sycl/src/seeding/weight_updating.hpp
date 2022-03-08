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
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/seeding/detail/triplet.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

namespace traccc::sycl {

/// Forward declaration of weight updating function
/// The weight of triplets are updated by iterating over triplets which share
/// the same middle spacepoint
///
/// @param config seed finder config
/// @param internal_sp_container vecmem container for internal spacepoint
/// @param triplet_counter_container vecmem container for triplet counters
/// @param triplet_container vecmem container for triplets
/// @param resource vecmem memory resource
/// @param q sycl queue for kernel scheduling
void weight_updating(const seedfilter_config& filter_config,
                     sp_grid& internal_sp,
                     host_triplet_counter_container& triplet_counter_container,
                     host_triplet_container& triplet_container,
                     vecmem::memory_resource& resource, queue_wrapper queue);

}  // namespace traccc::sycl
