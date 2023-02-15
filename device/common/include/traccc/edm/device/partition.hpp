/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// traccc include(s).
#include "traccc/edm/container.hpp"

namespace traccc::device {

/// A partitioning point used for dividing collections into similarly sized
/// fragments
/// A partitioning vector should have 0 as the first element and the size of the
/// collection to be partitioned as the last element, with the partitioning
/// points in the middle sorted.
using partition = unsigned int;

/// Declare all partition collection types
using partition_collection_types = collection_types<partition>;

}  // namespace traccc::device
