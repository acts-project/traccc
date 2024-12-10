/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "vecmem/memory/allocator.hpp"

#include "vecmem/memory/memory_resource.hpp"

namespace vecmem {
allocator::allocator(memory_resource& mem) : m_mem(mem) {}

void* allocator::allocate_bytes(std::size_t bytes, std::size_t alignment) {

    if (bytes == 0) {
        return nullptr;
    }

    /*
     * This allocation method simply wraps the upstream allocator's allocate
     * method. Nothing special here.
     */
    return m_mem.allocate(bytes, alignment);
}

void allocator::deallocate_bytes(void* p, std::size_t bytes,
                                 std::size_t alignment) {

    if (p == nullptr) {
        return;
    }

    /*
     * Again, we just wrap the upstream deallocate method.
     */
    m_mem.deallocate(p, bytes, alignment);
}
}  // namespace vecmem
