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
#include <functional>
#include <unordered_map>
#include <vector>

namespace vecmem::details {

/// Implementation for @c vecmem::details::coalescing_memory_resource
class coalescing_memory_resource_impl {

public:
    /// Constructor
    coalescing_memory_resource_impl(
        std::vector<std::reference_wrapper<memory_resource>>&& upstreams);

    /// Allocate memory with a chosen memory resource
    void* allocate(std::size_t size, std::size_t align);
    /// Deallocate previously allocated memory
    void deallocate(void* p, std::size_t size, std::size_t align);

private:
    /// The vector of upstream memory resources
    const std::vector<std::reference_wrapper<memory_resource>> m_upstreams;

    /// The map of allocations to memory resources
    std::unordered_map<void*, std::reference_wrapper<memory_resource>>
        m_allocations;

};  // class coalescing_memory_resource_impl

}  // namespace vecmem::details
