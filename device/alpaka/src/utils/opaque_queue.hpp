/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "utils.hpp"

namespace traccc::alpaka::details {

/// RAII wrapper around @c ::alpaka::Queue
///
/// It is used only internally by the Alpaka library, so it does not need to
/// provide any nice interface.
///
struct opaque_queue {

    /// Default constructor
    opaque_queue(std::size_t device);

    /// Device that the queue is associated to
    std::size_t m_device;
    /// Queue managed by the object
    std::unique_ptr<Queue> m_queue;
    /// Device-specific queue object
    void* m_nativeQueue;

};  // class opaque_queue

}  // namespace traccc::alpaka::details
