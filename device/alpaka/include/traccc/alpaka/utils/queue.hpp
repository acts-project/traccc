/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <cstddef>
#include <limits>
#include <memory>

namespace traccc::alpaka {

// Forward declaration(s).
namespace details {
struct opaque_queue;
}

/// Owning wrapper class around @c ::alpaka::Queue
///
/// It is necessary for passing around Alpaka queue objects in code that should
/// not be directly exposed to the Alpaka header(s).
///
class queue {

    public:
    /// Invalid/default device identifier
    static constexpr std::size_t INVALID_DEVICE =
        std::numeric_limits<std::size_t>::max();

    /// Construct a new queue (possibly for a specified device)
    queue(std::size_t device = INVALID_DEVICE);

    /// Move constructor
    queue(queue&& parent);

    /// Destructor
    ~queue();

    /// Move assignment
    queue& operator=(queue&& rhs);

    /// Device that the queue is associated to
    std::size_t device() const;

    /// Wait for all queued tasks from the queue to complete
    void synchronize() const;

    /// Access a typeless pointer to the managed @c ::alpaka::Queue object
    void* alpakaQueue() const;

    /// Access a typeless pointer to the underlying, device-specific queue
    /// object I.e. @c cudaStream_t for CUDA, @c hipStream_t for HIP, etc.
    void* deviceNativeQueue() const;

    private:
    /// Smart pointer to the managed @c ::alpaka::Queue object
    std::unique_ptr<details::opaque_queue> m_queue;

};  // class queue

}  // namespace traccc::alpaka
