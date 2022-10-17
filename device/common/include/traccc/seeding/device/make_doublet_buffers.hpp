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
#include "traccc/seeding/detail/doublet.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

namespace traccc::device {

/// Helper struct for the return type of @c traccc::device::make_doublet_buffer
struct doublet_buffer_pair {
    /// Middle-bottom doublet buffer
    doublet_container_types::buffer middleBottom;
    /// Middle-top doublet buffer
    doublet_container_types::buffer middleTop;
};

/// Create the doublet buffers
///
/// This is a helper function to use for setting up the buffers that
/// @c traccc::device::find_doublets would be able to fill.
///
/// @param doublet_counter The doublet counter from
///                        @c traccc::device::count_doublets
/// @param copy The object to use for accessing @c doublet_counter
/// @param mr The memory resource for the buffer
/// @param mr_host The (optional) host-accessible memory resource for the
///                buffer, in case @c mr is not host-accessible
/// @return Buffers usable by @c traccc::device::find_doublets
///
doublet_buffer_pair make_doublet_buffers(
    const doublet_counter_container_types::const_view& doublet_counter,
    vecmem::copy& copy, vecmem::memory_resource& mr,
    vecmem::memory_resource* mr_host = nullptr);

}  // namespace traccc::device
