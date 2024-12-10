/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "coalescing_memory_resource_impl.hpp"

// System include(s).
#include <cassert>

namespace vecmem::details {

coalescing_memory_resource_impl::coalescing_memory_resource_impl(
    std::vector<std::reference_wrapper<memory_resource>> &&upstreams)
    : m_upstreams(upstreams) {}

void *coalescing_memory_resource_impl::allocate(std::size_t size,
                                                std::size_t align) {

    if (size == 0) {
        return nullptr;
    }

    /*
     * Try to allocate with each of the upstream resources.
     */
    for (memory_resource &res : m_upstreams) {
        try {
            /*
             * Try to allocate the memory, and store the result with the
             * allocator reference in the allocation map.
             */
            void *ptr = res.allocate(size, align);

            m_allocations.emplace(ptr, res);

            return ptr;
        } catch (std::bad_alloc &) {
            /*
             * If we cannot allocate with this resource, try the next one.
             */
            continue;
        }
    }

    /*
     * If all resources fail to allocate, then we do as well.
     */
    throw std::bad_alloc();
}

void coalescing_memory_resource_impl::deallocate(void *ptr, std::size_t size,
                                                 std::size_t align) {

    if (ptr == nullptr) {
        return;
    }

    /*
     * Fetch the resource used to allocate this pointer.
     */
    auto nh = m_allocations.extract(ptr);

    /*
     * For debug builds, throw an assertion error if we do not know this
     * allocation.
     */
    assert(nh);

    /*
     * If we know who allocated this memory, forward the deallocation request
     * to them.
     */
    memory_resource &res = nh.mapped();

    res.deallocate(nh.key(), size, align);
}

}  // namespace vecmem::details
