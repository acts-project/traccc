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
#include "triplet_counter.hpp"

// Project include(s).
#include "traccc/edm/seed.hpp"
#include "traccc/seeding/common/seeding_config.hpp"
#include "traccc/seeding/common/spacepoint_grid.hpp"
#include "traccc/seeding/common/triplet.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/memory/memory_resource.hpp>

namespace traccc::sycl {

/// Forward declaration of seed selecting function
/// The good triplets are selected and recorded into seed container
///
/// @param filter_config seed filter config
/// @param internal_sp_container vecmem container for internal spacepoint
/// @param doublet_counter_container vecmem container for doublet_counter
/// @param triplet_counter_container vecmem container for triplet counters
/// @param triplet_container vecmem container for triplets
/// @param seed_container vecmem container for seeds
/// @param resource vecmem memory resource
/// @param q sycl queue for kernel scheduling
void seed_selecting(const seedfilter_config& filter_config,
                    host_spacepoint_container& spacepoints,
                    sp_grid& internal_sp,
                    host_doublet_counter_container& doublet_counter_container,
                    host_triplet_counter_container& triplet_counter_container,
                    host_triplet_container& triplet_container,
                    vecmem::data::vector_buffer<seed>& seed_buffer,
                    vecmem::memory_resource& resource, queue_wrapper queue);

}  // namespace traccc::sycl
