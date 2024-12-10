/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "conditional_memory_resource_impl.hpp"

namespace vecmem::details {

conditional_memory_resource_impl::conditional_memory_resource_impl(
    memory_resource &upstream,
    std::function<bool(std::size_t, std::size_t)> pred)
    : m_upstream(upstream), m_pred(pred) {}

void *conditional_memory_resource_impl::allocate(std::size_t size,
                                                 std::size_t align) {

    if (size == 0) {
        return nullptr;
    }

    /*
     * Check whether the predicate is true for this allocation. If it is, we
     * allocate as normal. If not, we refuse the allocation.
     */
    if (m_pred(size, align)) {
        return m_upstream.allocate(size, align);
    } else {
        throw std::bad_alloc();
    }
}

void conditional_memory_resource_impl::deallocate(void *ptr, std::size_t size,
                                                  std::size_t align) {

    if (ptr == nullptr) {
        return;
    }

    /*
     * The deallocation is a simple forwarding method, since we live under the
     * assumption that clients will never ask us deallocate memory that we
     * failed to allocate!
     */
    m_upstream.deallocate(ptr, size, align);
}

}  // namespace vecmem::details
