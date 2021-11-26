/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <CL/sycl.hpp>

#include <algorithm>
#include "sycl/seeding/detail/doublet_counter.hpp"
#include "sycl/seeding/detail/triplet_counter.hpp"
#include "sycl/seeding/detail/sycl_helper.hpp"
#include <edm/internal_spacepoint.hpp>
#include <edm/seed.hpp>
#include <seeding/detail/doublet.hpp>
#include <seeding/detail/seeding_config.hpp>
#include <seeding/detail/triplet.hpp>
#include <seeding/seed_selecting_helper.hpp>

#pragma once

namespace traccc {
namespace sycl {


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
                    host_internal_spacepoint_container& internal_sp_container,
                    host_doublet_counter_container& doublet_counter_container,
                    host_triplet_counter_container& triplet_counter_container,
                    host_triplet_container& triplet_container,
                    host_seed_container& seed_container,
                    vecmem::memory_resource* resource,
                    ::sycl::queue* q);

// Short aliast for accessor to local memory (shared memory in CUDA)
template <typename T>
using local_accessor = ::sycl::accessor<
    T,
    1,
    ::sycl::access::mode::read_write,
    ::sycl::access::target::local>;
    
}  // namespace sycl
}  // namespace traccc
