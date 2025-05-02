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

    /// Constructor
    /// @param device The device to create the queue for
    explicit impl(std::size_t device)
        : m_device(device == INVALID_DEVICE ? 0 : device),
          m_queue(::alpaka::getDevByIdx(::alpaka::Platform<Acc>{}, m_device)) {}

    /// The device the queue is created for
    std::size_t m_device;

    /// The real Alpaka queue object
    Queue m_queue;

};  // struct queue::impl

queue::queue(std::size_t device) : m_impl{std::make_unique<impl>(device)} {}

queue::queue(queue&&) noexcept = default;

queue::~queue() = default;

queue& queue::operator=(queue&& rhs) noexcept = default;

std::size_t queue::device() const {

    return m_impl->m_device;
}

void* queue::alpakaQueue() {

    return &(m_impl->m_queue);
}

const void* queue::alpakaQueue() const {

    return &(m_impl->m_queue);
}

void queue::synchronize() {

    ::alpaka::wait(m_impl->m_queue);
}

}  // namespace traccc::alpaka
