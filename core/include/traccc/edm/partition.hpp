/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// traccc include(s).
#include "traccc/edm/container.hpp"

namespace traccc {

namespace partitioning {
// Define number of cells to put together in each partition. Equal to the number
// of threads per block in the CCL kernel.
static constexpr std::size_t MAX_CELLS_PER_PARTITION = 1024;
}  // namespace partitioning

/// Partition used for dividing collections into similarly sized fragments
using partition = unsigned int;

/// Declare all partition collection types
using partition_collection_types = collection_types<partition>;

}  // namespace traccc
