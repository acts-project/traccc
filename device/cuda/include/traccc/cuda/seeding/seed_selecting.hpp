/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/cuda/seeding/detail/triplet_counter.hpp"
#include "traccc/edm/device/doublet_counter.hpp"
#include "traccc/edm/internal_spacepoint.hpp"
#include "traccc/edm/seed.hpp"
#include "traccc/seeding/detail/doublet.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/seeding/detail/triplet.hpp"
#include "traccc/seeding/seed_selecting_helper.hpp"

namespace traccc {
namespace cuda {

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
void seed_selecting(
    const seedfilter_config& filter_config,
    const vecmem::vector<device::doublet_counter_header>& dcc_headers,
    const spacepoint_container_types::view& spacepoints,
    sp_grid_const_view internal_sp_view,
    device::doublet_counter_container_types::const_view dcc_view,
    triplet_counter_container_view tcc_view, triplet_container_view tc_view,
    vecmem::data::vector_buffer<seed>& seed_buffer,
    vecmem::memory_resource& resource);

}  // namespace cuda
}  // namespace traccc
