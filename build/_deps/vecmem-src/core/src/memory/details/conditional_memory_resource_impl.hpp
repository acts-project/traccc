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
#include <functional>

namespace vecmem::details {

/// Implementation for @c vecmem::details::conditional_memory_resource
class conditional_memory_resource_impl {

public:
    /// Constructor
    conditional_memory_resource_impl(
        memory_resource& upstream,
        std::function<bool(std::size_t, std::size_t)> pred);

    /// (Conditionally) Allocate memory with a chosen memory resource
    void* allocate(std::size_t, std::size_t);
    /// Deallocate previously allocated memory
    void deallocate(void* p, std::size_t, std::size_t);

private:
    /// The upstream memory resource
    memory_resource& m_upstream;

    /// The predicate function
    std::function<bool(std::size_t, std::size_t)> m_pred;

};  // class conditional_memory_resource_impl

}  // namespace vecmem::details
