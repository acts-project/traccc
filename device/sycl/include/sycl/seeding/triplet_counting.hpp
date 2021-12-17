/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

#include <CL/sycl.hpp>

#include "sycl/seeding/detail/doublet_counter.hpp"
#include "sycl/seeding/detail/triplet_counter.hpp"
#include "sycl/seeding/detail/sycl_helper.hpp"
#include <vecmem/memory/atomic.hpp>
#include <edm/internal_spacepoint.hpp>
#include <seeding/detail/doublet.hpp>
#include <seeding/detail/seeding_config.hpp>
#include <seeding/detail/triplet.hpp>
#include <seeding/doublet_finding_helper.hpp>
#include <seeding/triplet_finding_helper.hpp>


namespace traccc {
namespace sycl {
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
/// @param resource vecmem memory resource
/// @param q sycl queue for kernel scheduling
void triplet_counting(const seedfinder_config& config,
                      host_internal_spacepoint_container& internal_sp_container,
                      host_doublet_counter_container& doublet_counter_container,
                      host_doublet_container& mid_bot_doublet_container,
                      host_doublet_container& mid_top_doublet_container,
                      host_triplet_counter_container& triplet_counter_container,
                      vecmem::memory_resource* resource,
                      ::sycl::queue* q);
    
} // namespace sycl
} // namespace traccc