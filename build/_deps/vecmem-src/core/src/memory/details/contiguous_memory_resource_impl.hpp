/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"

// System include(s).
#include <cstddef>

namespace vecmem::details {

/// Implementation of @c vecmem::contiguous_memory_resource
class contiguous_memory_resource_impl {

public:
    /// Constructor
    contiguous_memory_resource_impl(memory_resource& upstream,
                                    std::size_t size);
    /// Destructor
    ~contiguous_memory_resource_impl();

    /// Allocate memory from the resource's contiguous memory block
    void* allocate(std::size_t size, std::size_t alignment);
    /// De-allocate a previously allocated memory block
    void deallocate(void* ptr, std::size_t size, std::size_t alignment);

private:
    /// Upstream memory resource to allocate the one memory blob with
    memory_resource& m_upstream;
    /// Size of memory to allocate upstream
    const std::size_t m_size;
    /// Pointer to the memory blob allocated from upstream
    void* const m_begin;
    /// Pointer to the next free memory block to give out
    void* m_next;

};  // class contiguous_memory_resource_impl

}  // namespace vecmem::details
