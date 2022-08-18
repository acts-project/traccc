/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/device/get_prefix_sum.hpp"

// System include(s).
#include <algorithm>
#include <cassert>

namespace traccc::device {

TRACCC_HOST
prefix_sum_t get_prefix_sum(
    const std::vector<vecmem::data::vector_view<int>::size_type>& sizes,
    vecmem::memory_resource& mr) {

    // Create the result object.
    const std::size_t nelements = static_cast<std::size_t>(
        std::accumulate(sizes.begin(), sizes.end(), 0));
    prefix_sum_t result(nelements, &mr);

    // Fill the result object.
    for (std::size_t i = 0, k = 0; i < sizes.size(); ++i) {
        for (std::size_t j = 0; j < sizes[i]; ++j, ++k) {
            assert(k < result.size());
            result[k] = {i, j};
        }
    }

    // Return the prefix sum.
    return result;
}

}  // namespace traccc::device
