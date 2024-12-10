/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "debug_memory_resource_impl.hpp"

// System include(s).
#include <sstream>
#include <stdexcept>

namespace vecmem::details {

debug_memory_resource_impl::debug_memory_resource_impl(
    memory_resource &upstream)
    : m_upstream(upstream) {}

void *debug_memory_resource_impl::allocate(std::size_t size,
                                           std::size_t align) {

    if (size == 0) {
        return nullptr;
    }

    /*
     * Forward the allocation upstream. At this time, we can't really check for
     * any errors yet.
     */
    void *ptr = m_upstream.allocate(size, align);

    /*
     * Calculate the end pointer of this allocation.
     */
    void *end = static_cast<void *>(static_cast<char *>(ptr) + size);

    /*
     * Search for any potentially overlapping outstanding allocations.
     */
    for (const std::pair<const void *, std::pair<std::size_t, std::size_t>> i :
         m_allocations) {
        const void *i_beg = i.first;
        std::size_t i_size = i.second.first;
        const void *i_end = static_cast<const void *>(
            static_cast<const char *>(i_beg) + i_size);

        if (ptr < i_end && i_beg < end) {
            std::stringstream msg;

            msg << "Allocation error: allocation at " << ptr << " (size "
                << size << ") overlaps with previous allocation " << i_beg
                << " (size " << i_size << ").";

            throw std::logic_error(msg.str());
        }
    }

    /*
     * Store the current allocation as an outstanding one.
     */
    m_allocations.emplace(ptr, std::pair(size, align));

    return ptr;
}

void debug_memory_resource_impl::deallocate(void *ptr, std::size_t size,
                                            std::size_t align) {

    if (ptr == nullptr) {
        return;
    }

    auto alloc_it = m_allocations.find(ptr);

    /*
     * Check whether we are aware of this allocation at all.
     */
    if (alloc_it == m_allocations.end()) {
        std::stringstream msg;

        msg << "Deallocation error: memory at " << ptr << " (size " << size
            << ") was not previously allocated (or has already been "
               "deallocated).";

        throw std::logic_error(msg.str());
    }

    const std::pair<std::size_t, std::size_t> &alloc = alloc_it->second;

    /*
     * Check whether the deallocation arguments match the allocation arguments.
     */
    if (alloc.first != size || alloc.second != align) {
        std::stringstream msg;

        msg << "Deallocation error: allocation at " << ptr
            << " exists, but size (" << size << " vs. " << alloc.first
            << ") or alignment (" << align << " vs. " << alloc.second
            << ") does not match.";

        throw std::logic_error(msg.str());
    }

    /*
     * After we confirm that this pointer was actually allocated with this
     * resource, we can continue by forwarding the request upstream.
     */
    m_upstream.deallocate(ptr, size, align);

    /*
     * Finally, we need to make sure that the allocation is removed from our
     * administration.
     */
    m_allocations.erase(alloc_it);
}

}  // namespace vecmem::details
