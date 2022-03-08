/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// SYCL library include(s).
#include "doublet_counter.hpp"
#include "traccc/sycl/utils/queue_wrapper.hpp"

// Project include(s).
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

namespace traccc::sycl {

/// Forward declaration of doublet counting function
/// The number of mid-bot and mid-top doublets are counted for all spacepoints
/// and recorded into doublet counter container if the number of doublets are
/// larger than zero.
///
/// @param config seed finder config
/// @param internal_sp_container vecmem container for internal spacepoint
/// @param doublet_counter_container vecmem container for doublet_counter
/// @param resource vecmem memory resource
/// @param q sycl queue for kernel scheduling
void doublet_counting(const seedfinder_config& config, sp_grid& internal_sp,
                      host_doublet_counter_container& doublet_counter_container,
                      vecmem::memory_resource& resource, queue_wrapper queue);

}  // namespace traccc::sycl
