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
#include <unordered_map>

namespace vecmem::details {

/// Implementation for @c vecmem::details::debug_memory_resource
class debug_memory_resource_impl {

public:
    /**
     * @brief Constructs the debug memory resource.
     *
     * @param[in] upstream The upstream memory resource to use.
     */
    debug_memory_resource_impl(memory_resource& upstream);

    /// Allocate memory with a chosen memory resource
    void* allocate(std::size_t, std::size_t);
    /// Deallocate previously allocated memory
    void deallocate(void* p, std::size_t, std::size_t);

private:
    /// The upstream memory resource
    memory_resource& m_upstream;
    /// Previously allocated memory blocks
    std::unordered_map<void*, std::pair<std::size_t, std::size_t>>
        m_allocations;

};  // class debug_memory_resource_impl

}  // namespace vecmem::details
