/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/device/triplet_counter.hpp"

// Project include(s).
#include "traccc/seeding/detail/triplet.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

namespace traccc::device {

/// Create the triplet buffer
///
/// This is a helper function to use for setting up the buffer that
/// @c traccc::device::find_triplets would be able to fill.
///
/// @param triplet_counter The triplet counter from
///                        @c traccc::device::count_triplets
/// @param copy The object to use for accessing @c triplet_counter
/// @param mr The memory resource for the buffer
/// @param mr_host The (optional) host-accessible memory resource for the
///                buffer, in case @c mr is not host-accessible
/// @return Buffer usable by @c traccc::device::find_triplets
///
triplet_container_buffer make_triplet_buffer(
    const triplet_counter_container_types::const_view& triplet_counter,
    vecmem::copy& copy, vecmem::memory_resource& mr,
    vecmem::memory_resource* mr_host = nullptr);

}  // namespace traccc::device
