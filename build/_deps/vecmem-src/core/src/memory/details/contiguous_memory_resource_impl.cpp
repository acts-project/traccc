/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "contiguous_memory_resource_impl.hpp"

#include "vecmem/utils/debug.hpp"

// System include(s).
#include <cassert>
#include <memory>
#include <stdexcept>

namespace vecmem::details {

contiguous_memory_resource_impl::contiguous_memory_resource_impl(
    memory_resource &upstream, std::size_t size)
    : m_upstream(upstream),
      m_size(size),
      m_begin(m_upstream.allocate(m_size)),
      m_next(m_begin) {

    VECMEM_DEBUG_MSG(
        2, "Allocated %lu bytes at %p from the upstream memory resource",
        m_size, m_begin);
}

contiguous_memory_resource_impl::~contiguous_memory_resource_impl() {
    /*
     * Deallocate our memory arena upstream.
     */
    m_upstream.deallocate(m_begin, m_size);
    VECMEM_DEBUG_MSG(
        2, "De-allocated %lu bytes at %p using the upstream memory resource",
        m_size, m_begin);
}

void *contiguous_memory_resource_impl::allocate(std::size_t size,
                                                std::size_t alignment) {

    if (size == 0) {
        return nullptr;
    }

    /*
     * Compute the remaining space, which needs to be an lvalue for standard
     * library-related reasons.
     */
    assert(m_next >= m_begin);
    std::size_t rem =
        m_size - static_cast<std::size_t>(static_cast<char *>(m_next) -
                                          static_cast<char *>(m_begin));

    /*
     * Employ std::align to find the next properly aligned address.
     */
    if (std::align(alignment, size, m_next, rem)) {
        /*
         * Store the return pointer, update the stored next pointer, then
         * return.
         */
        void *res = m_next;
        m_next = static_cast<char *>(m_next) + size;

        VECMEM_DEBUG_MSG(4, "Allocated %lu bytes at %p", size, res);

        return res;
    } else {
        /*
         * If std::align returns a false-like value, the allocation has failed
         * and we throw an exception.
         */
        throw std::bad_alloc();
    }
}

void contiguous_memory_resource_impl::deallocate(void *, std::size_t,
                                                 std::size_t) {
    /*
     * Deallocation is a no-op for this memory resource, so we do nothing.
     */
    return;
}

}  // namespace vecmem::details
