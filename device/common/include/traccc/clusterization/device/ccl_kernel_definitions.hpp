/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device::details {
/// These indices in clusterization will only range from 0 to
/// max_cells_per_partition, so we only need a short
using index_t = unsigned short;

/// The limit on the stack size in terms of cells per thread.
static constexpr std::size_t CELLS_PER_THREAD_STACK_LIMIT = 32;
}  // namespace traccc::device::details
