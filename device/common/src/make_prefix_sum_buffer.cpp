/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/device/make_prefix_sum_buffer.hpp"

// System include(s).
#include <algorithm>

namespace traccc::device {

TRACCC_HOST
prefix_sum_buffer_t make_prefix_sum_buffer(
    const std::vector<prefix_sum_size_t>& sizes, vecmem::copy& copy,
    const traccc::memory_resource& mr) {

    if (sizes.size() == 0) {
        return {{}, {}, 0};
    }

    // Create vector with summation of sizes
    vecmem::vector<prefix_sum_size_t> sizes_sum(sizes.size(),
                                                mr.host ? mr.host : &(mr.main));
    std::partial_sum(sizes.begin(), sizes.end(), sizes_sum.begin(),
                     std::plus<prefix_sum_size_t>());
    const prefix_sum_size_t totalSize = sizes_sum.back();

    if (mr.host != nullptr) {
        // Create buffer and view objects
        vecmem::data::vector_buffer<prefix_sum_size_t> sizes_sum_buff(
            sizes_sum.size(), mr.main);
        copy.setup(sizes_sum_buff);
        (copy)(vecmem::get_data(sizes_sum), sizes_sum_buff)->wait();
        vecmem::data::vector_view<prefix_sum_size_t> sizes_sum_view(
            sizes_sum_buff);

        return {sizes_sum_view, std::move(sizes_sum_buff), totalSize};
    } else {
        // Create view object
        vecmem::data::vector_view<prefix_sum_size_t> sizes_sum_view =
            vecmem::get_data(sizes_sum);

        return {sizes_sum_view, std::move(sizes_sum), totalSize};
    }
}

}  // namespace traccc::device
