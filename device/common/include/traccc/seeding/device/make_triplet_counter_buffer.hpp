/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/device/triplet_counter.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// System include(s)
#include <cstddef>
#include <vector>

namespace traccc::device {

/// Create a triplet counter buffer
///
/// This is a helper function to use for setting up the buffer that
/// @c traccc::device::count_triplets would be able to fill.
///
/// @param mb_doublet_sizes vector of MidBot doublet sizes
/// @param copy The object to use for accessing @c spacepoint_grid
/// @param mr The memory resource for the buffer
/// @param mr_host The (optional) host-accessible memory resource for the
///                buffer, in case @c mr is not host-accessible
/// @return A buffer usable by @c traccc::device::count_triplets
///
triplet_counter_container_types::buffer make_triplet_counter_buffer(
    const std::vector<std::size_t>& mb_doublet_sizes, vecmem::copy& copy,
    vecmem::memory_resource& mr, vecmem::memory_resource* mr_host = nullptr);

}  // namespace traccc::device
