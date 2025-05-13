/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/alpaka/utils/queue.hpp"

#include "utils.hpp"

// Alpaka include(s).
#include <alpaka/alpaka.hpp>

namespace traccc::alpaka {

struct queue::impl {
    /// The device the queue is created for
    std::size_t m_device{INVALID_DEVICE};
    /// Bare pointer to the wrapped queue object
    Queue* m_queue = nullptr;
    /// Unique pointer to the managed Queue object
    std::unique_ptr<Queue> m_managedQueue;
};  // struct queue::impl

queue::queue(std::size_t device) : m_impl{std::make_unique<impl>()} {

    m_impl->m_device = device == INVALID_DEVICE ? 0 : device;
    m_impl->m_managedQueue = std::make_unique<Queue>(
        ::alpaka::getDevByIdx(::alpaka::Platform<Acc>{}, m_impl->m_device));
    m_impl->m_queue = m_impl->m_managedQueue.get();
}

queue::queue(void* input_queue) : m_impl{std::make_unique<impl>()} {

    assert(input_queue != nullptr);
    m_impl->m_queue = static_cast<Queue*>(input_queue);

#if defined(ALPAKA_ACC_SYCL_ENABLED)
    auto sycl_device =
        ::alpaka::getNativeHandle(::alpaka::getDev(*m_impl->m_queue)).first;
    m_impl->m_device = static_cast<std::size_t>(
        sycl_device.get_info<::sycl::info::device::vendor_id>());
#else
    m_impl->m_device = static_cast<std::size_t>(
        ::alpaka::getNativeHandle(::alpaka::getDev(*m_impl->m_queue)));
#endif
}

queue::queue(queue&&) noexcept = default;

queue::~queue() = default;

queue& queue::operator=(queue&& rhs) noexcept = default;

std::size_t queue::device() const {

    return m_impl->m_device;
}

void* queue::alpakaQueue() {

    assert(m_impl->m_queue != nullptr);
    return m_impl->m_queue;
}

const void* queue::alpakaQueue() const {

    assert(m_impl->m_queue != nullptr);
    return m_impl->m_queue;
}

void queue::synchronize() {

    assert(m_impl->m_queue != nullptr);
    ::alpaka::wait(*m_impl->m_queue);
}

}  // namespace traccc::alpaka
