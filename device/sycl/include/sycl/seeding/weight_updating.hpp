/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <CL/sycl.hpp>

#include "sycl/seeding/detail/triplet_counter.hpp"
#include "sycl/seeding/detail/sycl_helper.hpp"
#include <edm/internal_spacepoint.hpp>
#include <seeding/detail/seeding_config.hpp>
#include <seeding/detail/triplet.hpp>

namespace traccc {
namespace sycl {

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
                     host_internal_spacepoint_container& internal_sp_container,
                     host_triplet_counter_container& triplet_counter_container,
                     host_triplet_container& triplet_container,
                     vecmem::memory_resource* resource,
                     ::sycl::queue* q);

}  // namespace sycl
}  // namespace traccc
