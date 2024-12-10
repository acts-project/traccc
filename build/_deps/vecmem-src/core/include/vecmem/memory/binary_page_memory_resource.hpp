/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/details/memory_resource_base.hpp"
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/vecmem_core_export.hpp"

// System include(s).
#include <memory>

namespace vecmem {

// Forward declaration(s).
namespace details {
struct binary_page_memory_resource_impl;
}

/**
 * @brief A memory manager using power-of-two pages that can be split to
 * deal with allocation requests of various sizes.
 *
 * This is a non-terminal memory resource which relies on an upstream
 * allocator to do the actual allocation. The allocator will allocate only
 * large blocks with sizes power of two from the upstream allocator. These
 * blocks can then be split in half and allocated, split in half again. This
 * creates a binary tree of pages which can be either vacant, occupied, or
 * split.
 */
class binary_page_memory_resource final : public details::memory_resource_base {

public:
    /**
     * @brief Initialize a binary page memory manager depending on an
     * upstream memory resource.
     */
    VECMEM_CORE_EXPORT
    binary_page_memory_resource(memory_resource&);
    /// Move constructor
    VECMEM_CORE_EXPORT
    binary_page_memory_resource(binary_page_memory_resource&& parent);
    /// Disallow copying the memory resource
    binary_page_memory_resource(const binary_page_memory_resource&) = delete;

    /**
     * @brief Deconstruct a binary page memory manager, freeing all
     * allocated blocks upstream.
     *
     * The destructor is explicitly implemented to not require clients of the
     * class to know how to destruct
     * @c vecmem::details::binary_page_memory_resource_impl.
     */
    VECMEM_CORE_EXPORT
    ~binary_page_memory_resource();

    /// Move assignment operator
    VECMEM_CORE_EXPORT
    binary_page_memory_resource& operator=(binary_page_memory_resource&& rhs);
    /// Disallow copying the memory resource
    binary_page_memory_resource& operator=(const binary_page_memory_resource&) =
        delete;

private:
    /// @name Function(s) implementing @c vecmem::memory_resource
    /// @{

    /// Allocate a blob of memory
    VECMEM_CORE_EXPORT
    virtual void* do_allocate(std::size_t, std::size_t) override final;
    /// De-allocate a previously allocated memory blob
    VECMEM_CORE_EXPORT
    virtual void do_deallocate(void* p, std::size_t,
                               std::size_t) override final;

    /// @}

    /// Object implementing the memory resource's logic
    std::unique_ptr<details::binary_page_memory_resource_impl> m_impl;

};  // class binary_page_memory_resource

}  // namespace vecmem
