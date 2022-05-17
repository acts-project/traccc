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
#include "traccc/edm/seed.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/seeding/detail/triplet.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/memory/memory_resource.hpp>

namespace traccc::sycl {

/// Forward declaration of seed selecting function
/// The good triplets are selected and recorded into seed container
///
/// @param filter_config seed filter config
/// @param dcc_headers The header vector for the doublet counts
/// @param internal_sp_container vecmem container for internal spacepoint
/// @param doublet_counter_container vecmem container for doublet_counter
/// @param triplet_counter_container vecmem container for triplet counters
/// @param triplet_container vecmem container for triplets
/// @param seed_container vecmem container for seeds
/// @param resource vecmem memory resource
/// @param q sycl queue for kernel scheduling
void seed_selecting(
    const seedfilter_config& filter_config,
    const vecmem::vector<device::doublet_counter_header>& dcc_headers,
    const spacepoint_container_const_view& spacepoints_view,
    const sp_grid_const_view& internal_sp,
    const device::doublet_counter_container_types::const_view&
        doublet_counter_container,
    triplet_counter_container_view tcc_view,
    triplet_container_view tc_view,
    vecmem::data::vector_buffer<seed>& seed_buffer,
    queue_wrapper queue);

}  // namespace traccc::sycl
