/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <algorithm>
#include <cassert>
#include <type_traits>

namespace traccc::device {

template <typename CONTAINER_TYPES>
container_h2d_copy_alg<CONTAINER_TYPES>::container_h2d_copy_alg(
    const memory_resource& mr, vecmem::copy& copy)
    : m_mr(mr), m_copy(copy) {}

template <typename CONTAINER_TYPES>
typename container_h2d_copy_alg<CONTAINER_TYPES>::output_type
container_h2d_copy_alg<CONTAINER_TYPES>::operator()(input_type input) const {

    // Get the sizes of the jagged vector. Remember that the input comes from
    // host accessible memory. (Using the vecmem::copy object is not a good idea
    // in this case.)
    assert(input.headers.size() == input.items.size());
    const typename std::remove_reference<typename std::remove_cv<
        input_type>::type>::type::header_vector::size_type size =
        input.headers.size();
    std::vector<std::size_t> sizes(size, 0);
    const typename CONTAINER_TYPES::const_device device(input);
    std::transform(device.get_items().begin(), device.get_items().end(),
                   sizes.begin(), [](const auto& vec) { return vec.size(); });

    // Create the output buffer with the correct sizes.
    output_type result{{size, m_mr.main}, {sizes, m_mr.main, m_mr.host}};

    // Set it up, and copy data into it.
    m_copy.setup(result.headers);
    m_copy.setup(result.items);
    m_copy(input.headers, result.headers, vecmem::copy::type::host_to_device);
    m_copy(input.items, result.items, vecmem::copy::type::host_to_device);

    // Return the created buffer.
    return result;
}

}  // namespace traccc::device
