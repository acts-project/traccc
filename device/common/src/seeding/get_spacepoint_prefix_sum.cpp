/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/seeding/device/get_spacepoint_prefix_sum.hpp"

namespace traccc::device {

TRACCC_HOST
get_spacepoint_prefix_sum_result_t get_spacepoint_prefix_sum(
    const host_spacepoint_container& spacepoints, vecmem::memory_resource& mr) {

    // Create the result object.
    get_spacepoint_prefix_sum_result_t result(&mr);
    result.reserve(spacepoints.total_size());

    // Fill the result object.
    for (host_spacepoint_container::item_vector::size_type i = 0;
         i < spacepoints.size(); ++i) {

        const host_spacepoint_container::item_vector::value_type::size_type
            n_items = spacepoints.get_items()[i].size();

        for (host_spacepoint_container::item_vector::value_type::size_type j =
                 0;
             j < n_items; ++j) {
            result.push_back({i, j});
        }
    }

    // Return the prefix sum vector.
    return result;
}

}  // namespace traccc::device
