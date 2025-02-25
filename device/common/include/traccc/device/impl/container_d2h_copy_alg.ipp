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
#include <vector>

namespace traccc::device {

template <typename CONTAINER_TYPES>
container_d2h_copy_alg<CONTAINER_TYPES>::container_d2h_copy_alg(
    const memory_resource& mr, vecmem::copy& deviceCopy,
    std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)), m_mr(mr), m_deviceCopy(deviceCopy) {}

template <typename CONTAINER_TYPES>
typename container_d2h_copy_alg<CONTAINER_TYPES>::output_type
container_d2h_copy_alg<CONTAINER_TYPES>::operator()(input_type input) const {

    // A sanity check.
    assert(input.headers.size() == input.items.size());

    // Decide what memory resource to use for the host container.
    vecmem::memory_resource* host_mr =
        (m_mr.host != nullptr) ? m_mr.host : &(m_mr.main);

    // Create a temporary buffer that will receive the device memory.
    const typename std::remove_reference<typename std::remove_cv<
        input_type>::type>::type::header_vector::size_type size =
        input.headers.size();
    std::vector<std::size_t> capacities(size, 0);
    std::transform(input.items.host_ptr(), input.items.host_ptr() + size,
                   capacities.begin(),
                   [](const auto& view) { return view.capacity(); });
    typename CONTAINER_TYPES::buffer hostBuffer{{size, *host_mr},
                                                {capacities, *host_mr}};
    vecmem::copy::event_type host_header_setup_event =
        m_hostCopy.setup(hostBuffer.headers);
    vecmem::copy::event_type host_item_setup_event =
        m_hostCopy.setup(hostBuffer.items);
    host_header_setup_event->wait();
    host_item_setup_event->wait();

    // Copy the device container into this temporary host buffer.
    vecmem::copy::event_type device_header_copy_event = m_deviceCopy(
        input.headers, hostBuffer.headers, vecmem::copy::type::device_to_host);
    vecmem::copy::event_type device_item_copy_event = m_deviceCopy(
        input.items, hostBuffer.items, vecmem::copy::type::device_to_host);

    // Create the result object, giving it the appropriate memory resource for
    // all of its elements.
    output_type result{size, host_mr};
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i].items =
            typename CONTAINER_TYPES::host::item_vector::value_type{host_mr};
    }

    // Wait for the D->H copies to finish.
    device_header_copy_event->wait();
    device_item_copy_event->wait();

    // Perform the H->H copy.
    vecmem::copy::event_type host_header_copy_event =
        m_hostCopy(hostBuffer.headers, result.get_headers());
    vecmem::copy::event_type host_item_copy_event =
        m_hostCopy(hostBuffer.items, result.get_items());
    host_header_copy_event->wait();
    host_item_copy_event->wait();

    // Return the host object.
    return result;
}

}  // namespace traccc::device
