/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/cuda/seeding/detail/triplet_counter.hpp"
#include "traccc/edm/internal_spacepoint.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/seeding/detail/triplet.hpp"

namespace traccc {
namespace cuda {

/// Forward declaration of weight updating function
/// The weight of triplets are updated by iterating over triplets which share
/// the same middle spacepoint
///
/// @param config seed finder config
/// @param internal_sp_container vecmem container for internal spacepoint
/// @param triplet_counter_container vecmem container for triplet counters
/// @param triplet_container vecmem container for triplets
/// @param resource vecmem memory resource
void weight_updating(const seedfilter_config& filter_config,
                     const vecmem::vector<triplet_per_bin>& tc_headers,
                     sp_grid_const_view internal_sp_view,
                     triplet_counter_container_view tcc_view,
                     triplet_container_view tc_view,
                     vecmem::memory_resource& resource);

}  // namespace cuda
}  // namespace traccc
