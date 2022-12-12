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

namespace partition {
// Define number of cells to put together in each partition. Equal to the number
// of threads per block in the CCL kernel.
static constexpr std::size_t MAX_CELLS_PER_PARTITION = 1024;
}  // namespace partition

/// CCL partition used for dividing cell collection into similarly sized
/// fragments
struct ccl_partition {
    unsigned int start;

    // The size of each partition will range from 0 to MAX_CELLS_PER_PARTITION
    unsigned short size;
};  // struct ccl_partition

/// Declare all ccl_partition collection types
using ccl_partition_collection_types = collection_types<ccl_partition>;

}  // namespace traccc
