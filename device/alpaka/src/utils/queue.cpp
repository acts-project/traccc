/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/alpaka/utils/queue.hpp"

#include "opaque_queue.hpp"
#include "traccc/alpaka/utils/get_vecmem_resource.hpp"
#include "utils.hpp"

namespace traccc::alpaka {

queue::queue(std::size_t device) {

    // Make sure that the queue is constructed on the correct device.
    std::size_t selected_device = device == INVALID_DEVICE ? 0 : device;

    // Construct the queue.
    m_queue = std::make_unique<details::opaque_queue>(selected_device);
}

queue::queue(queue&& parent) : m_queue(std::move(parent.m_queue)) {}

/// The destructor is implemented explicitly to avoid clients of the class
/// having to know how to destruct @c traccc::alpaka::details::opaque_queue.
queue::~queue() {}

queue& queue::operator=(queue&& rhs) {

    // Avoid self-assignment.
    if (this == &rhs) {
        return *this;
    }

    // Move the managed queue object.
    m_queue = std::move(rhs.m_queue);

    // Return this object.
    return *this;
}

std::size_t queue::device() const {

    return m_queue->m_device;
}

void queue::synchronize() const {

    ::alpaka::wait(*(m_queue->m_queue));
}

void* queue::alpakaQueue() const {

    return static_cast<void*>(m_queue->m_queue.get());
}

void* queue::deviceNativeQueue() const {

#if defined(TRACCC_BUILD_CUDA) || defined(TRACCC_BUILD_HIP)
    return static_cast<void*>(::alpaka::getNativeHandle(*(m_queue->m_queue)));
#elif defined(TRACCC_BUILD_SYCL)
    auto nativeQueue = ::alpaka::getNativeHandle(*(m_queue->m_queue));
    return static_cast<void*>(&nativeQueue);
#else
    // TODO: What is the best way to handle this?
    //       Not having the method is a pain for other parts of the code,
    //       but having it and throwing an error is not great either.
    throw std::runtime_error("Native queue not available for this device");
#endif
}

}  // namespace traccc::alpaka
