/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/device/doublet_counter.hpp"

// Project include(s).
#include "traccc/seeding/detail/spacepoint_grid.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

namespace traccc::device {

/// Create a doublet counter buffer
///
/// This is a helper function to use for setting up the buffer that
/// @c traccc::device::count_doublets would be able to fill.
///
/// @param grid_sizes vector of grid sizes
/// @param copy The object to use for accessing @c spacepoint_grid
/// @param mr The memory resource for the buffer
/// @param mr_host The (optional) host-accessible memory resource for the
///                buffer, in case @c mr is not host-accessible
/// @return A buffer usable by @c traccc::device::count_doublets
///
doublet_counter_container_types::buffer make_doublet_counter_buffer(
    const std::vector<unsigned int>& grid_sizes, vecmem::copy& copy,
    vecmem::memory_resource& mr, vecmem::memory_resource* mr_host = nullptr);

}  // namespace traccc::device
