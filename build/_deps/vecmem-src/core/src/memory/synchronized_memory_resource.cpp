/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/synchronized_memory_resource.hpp"

namespace vecmem {

synchronized_memory_resource::synchronized_memory_resource(
    memory_resource& upstream)
    : m_upstream(upstream) {}

synchronized_memory_resource::synchronized_memory_resource(
    synchronized_memory_resource&& parent)
    : m_upstream(std::move(parent.m_upstream)) {}

synchronized_memory_resource::synchronized_memory_resource(
    const synchronized_memory_resource& parent)
    : m_upstream(parent.m_upstream) {}

synchronized_memory_resource::~synchronized_memory_resource() = default;

synchronized_memory_resource& synchronized_memory_resource::operator=(
    synchronized_memory_resource&& rhs) {

    if (this != &rhs) {
        m_upstream = std::move(rhs.m_upstream);
    }
    return *this;
}

synchronized_memory_resource& synchronized_memory_resource::operator=(
    const synchronized_memory_resource& rhs) {

    if (this != &rhs) {
        m_upstream = rhs.m_upstream;
    }
    return *this;
}

void* synchronized_memory_resource::do_allocate(std::size_t bytes,
                                                std::size_t alignment) {

    const std::lock_guard<std::mutex> lock(m_mutex);
    return m_upstream.get().allocate(bytes, alignment);
}

void synchronized_memory_resource::do_deallocate(void* p, std::size_t bytes,
                                                 std::size_t alignment) {

    const std::lock_guard<std::mutex> lock(m_mutex);
    m_upstream.get().deallocate(p, bytes, alignment);
}

bool synchronized_memory_resource::do_is_equal(
    const memory_resource& other) const noexcept {

    // Check if the other resource is also a synchronized resource.
    const synchronized_memory_resource* otherSynchronized =
        dynamic_cast<const synchronized_memory_resource*>(&other);
    if (otherSynchronized) {
        // If so, check if the underlying resources are equal.
        return m_upstream.get().is_equal(otherSynchronized->m_upstream.get());
    } else {
        // If not, check if the underlying resource is equal to the other
        // resource.
        return m_upstream.get().is_equal(other);
    }
}

}  // namespace vecmem
