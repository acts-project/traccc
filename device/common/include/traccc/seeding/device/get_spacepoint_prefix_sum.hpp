/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/spacepoint.hpp"

// VecMem include(s).
#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <utility>

namespace traccc::device {

/// Convenience type definition for the elements of the prefix sum
typedef std::pair<host_spacepoint_container::item_vector::size_type,
                  host_spacepoint_container::item_vector::value_type::size_type>
    get_spacepoint_prefix_sum_element_t;
/// Convenience type definition for the return value of the helper function
typedef vecmem::vector<get_spacepoint_prefix_sum_element_t>
    get_spacepoint_prefix_sum_result_t;

/// Construct a prefix sum that could be used to iterate over all spacepoints
///
/// Spacepoints are stored in a jagged vector in the EDM, separated for the
/// detector modules. In order to iterate over all of them in a 1-dimensional
/// loop, we use the output of this helper function.
///
/// @param spacepoints The spacepoints to create the prefix sum for
/// @return A 1-dimensional vector specifying how to iterate over the
/// spacepoints
///
TRACCC_HOST
get_spacepoint_prefix_sum_result_t get_spacepoint_prefix_sum(
    const host_spacepoint_container& spacepoints, vecmem::memory_resource& mr);

}  // namespace traccc::device
