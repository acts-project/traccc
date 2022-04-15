/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

template <typename element_t>
TRACCC_HOST prefix_sum_t
get_prefix_sum(const vecmem::data::jagged_vector_view<element_t>& view,
               vecmem::memory_resource& mr, vecmem::copy& copy) {

    return get_prefix_sum(copy.get_sizes(view), mr);
}

template <typename element_t>
TRACCC_HOST prefix_sum_t
get_prefix_sum(const vecmem::data::jagged_vector_buffer<element_t>& buffer,
               vecmem::memory_resource& mr, vecmem::copy& copy) {

    return get_prefix_sum(copy.get_sizes(buffer), mr);
}

}  // namespace traccc::device
