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

namespace traccc::device {

template <typename CONTAINER_TYPES>
container_h2d_copy_alg<CONTAINER_TYPES>::container_h2d_copy_alg(
    const memory_resource& mr, vecmem::copy& deviceCopy)
    : m_mr(mr), m_deviceCopy(deviceCopy), m_hostCopy() {}

template <typename CONTAINER_TYPES>
typename container_h2d_copy_alg<CONTAINER_TYPES>::output_type
container_h2d_copy_alg<CONTAINER_TYPES>::operator()(input_type input) const {

    // Get the sizes of the jagged vector.
    const std::vector<std::size_t> sizes = get_sizes(input);

    // Create the output buffer with the correct sizes.
    output_type result{{static_cast<header_size_type>(sizes.size()), m_mr.main},
                       {sizes, m_mr.main, m_mr.host}};
    m_deviceCopy.setup(result.headers);
    m_deviceCopy.setup(result.items);

    // Copy data straight into it.
    m_deviceCopy(input.headers, result.headers,
                 vecmem::copy::type::host_to_device);
    m_deviceCopy(input.items, result.items, vecmem::copy::type::host_to_device);

    // Return the created buffer.
    return result;
}

template <typename CONTAINER_TYPES>
typename container_h2d_copy_alg<CONTAINER_TYPES>::output_type
container_h2d_copy_alg<CONTAINER_TYPES>::operator()(
    input_type input, typename CONTAINER_TYPES::buffer& hostBuffer) const {

    // Get the sizes of the jagged vector.
    const std::vector<std::size_t> sizes = get_sizes(input);
    const header_size_type size = static_cast<header_size_type>(sizes.size());

    // Decide what memory resource to use for the host container.
    vecmem::memory_resource* host_mr =
        (m_mr.host != nullptr) ? m_mr.host : &(m_mr.main);

    // Create/set the host buffer.
    hostBuffer =
        typename CONTAINER_TYPES::buffer{{size, *host_mr}, {sizes, *host_mr}};
    m_hostCopy.setup(hostBuffer.headers);
    m_hostCopy.setup(hostBuffer.items);

    // Copy the data into the host buffer.
    m_hostCopy(input.headers, hostBuffer.headers);
    m_hostCopy(input.items, hostBuffer.items);

    // Create the output buffer with the correct sizes.
    output_type result{{size, m_mr.main}, {sizes, m_mr.main, m_mr.host}};
    m_deviceCopy.setup(result.headers);
    m_deviceCopy.setup(result.items);

    // Copy data from the host buffer into the device/result buffer.
    m_deviceCopy(hostBuffer.headers, result.headers,
                 vecmem::copy::type::host_to_device);
    m_deviceCopy(hostBuffer.items, result.items,
                 vecmem::copy::type::host_to_device);

    // Return the created buffer.
    return result;
}

template <typename CONTAINER_TYPES>
std::vector<std::size_t> container_h2d_copy_alg<CONTAINER_TYPES>::get_sizes(
    input_type input) const {

    // Get the sizes of the jagged vector. Remember that the input comes from
    // host accessible memory. (Using the vecmem::copy object is not a good idea
    // in this case.)
    assert(input.headers.size() == input.items.size());
    std::vector<std::size_t> sizes(input.headers.size(), 0);
    const typename CONTAINER_TYPES::const_device device(input);
    std::transform(device.get_items().begin(), device.get_items().end(),
                   sizes.begin(), [](const auto& vec) { return vec.size(); });

    // Return the sizes.
    return sizes;
}

}  // namespace traccc::device
