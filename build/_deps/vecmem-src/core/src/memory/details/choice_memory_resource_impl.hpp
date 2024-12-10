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

namespace vecmem::details {

/// Implementation for @c vecmem::details::choice_memory_resource
class choice_memory_resource_impl {

public:
    /// Constructor
    choice_memory_resource_impl(
        std::function<memory_resource&(std::size_t, std::size_t)> decision);

    /// Allocate memory with a chosen memory resource
    void* allocate(std::size_t size, std::size_t align);
    /// Deallocate previously allocated memory
    void deallocate(void* p, std::size_t size, std::size_t align);

private:
    /// The map of allocations to memory resources
    std::unordered_map<void*, std::reference_wrapper<memory_resource>>
        m_allocations;

    /// The function which picks the upstream memory resource to use
    std::function<memory_resource&(std::size_t, std::size_t)> m_decision;

};  // class choice_memory_resource_impl

}  // namespace vecmem::details
