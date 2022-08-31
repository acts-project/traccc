/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

TRACCC_HOST_DEVICE
void fill_prefix_sum(
    const std::size_t globalIndex,
    const vecmem::data::vector_view<const prefix_sum_size_t>& sizes_view,
    vecmem::data::vector_view<prefix_sum_element_t> ps_view) {

    const vecmem::device_vector<const prefix_sum_size_t> sizes(sizes_view);
    vecmem::device_vector<prefix_sum_element_t> result(ps_view);

    if (globalIndex >= sizes.size()) {
        return;
    }

    const prefix_sum_size_t previous =
        (globalIndex == 0) ? 0 : sizes[globalIndex - 1];
    const prefix_sum_size_t current = sizes[globalIndex];
    for (prefix_sum_size_t i = 0; i < current - previous; ++i) {
        result.at(previous + i) = {globalIndex, i};
    }
}

}  // namespace traccc::device
