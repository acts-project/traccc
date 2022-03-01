/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// SYCL library include(s).
#include "traccc/sycl/utils/queue_wrapper.hpp"

// Project include(s).
#include "traccc/edm/spacepoint.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"

// VecMem include(s).
#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <utility>

namespace traccc::sycl {

/// Forward declaration of populating grid function
/// This kernel populates the internal spacepoints into grid
///
/// Thread-block policy:
/// 1. Each thread is for each spacepoint
/// 2. The number of block is number of spacepoints / number of threads + 1
/// Therefore, each block may handle spacepoints from several modules
///
/// @param config seed finder config
/// @param sp_grid grid object to populate internal spacepoints
/// @param spacepoints spacepoint container
/// @param sp_container_indices header and item index of spacepoints
/// @param resource vecmem memory resource
/// @param q sycl queue for kernel scheduling
void populating_grid(
    const seedfinder_config config, sp_grid& grid,
    host_spacepoint_container& spacepoints,
    const vecmem::vector<std::pair<unsigned int, unsigned int>>&
        sp_container_indices,
    vecmem::vector<unsigned int>& grid_capacities,
    vecmem::memory_resource& resource, queue_wrapper queue);

}  // namespace traccc::sycl
