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
class debug_memory_resource_impl;
}

/**
 * @brief This memory resource forwards allocation and deallocation requests to
 * the upstream resource, but alerts the user of potential problems.
 *
 * For example, this memory resource can be used to catch overlapping
 * allocations, double frees, invalid frees, and other memory integrity issues.
 */
class debug_memory_resource final : public details::memory_resource_base {

public:
    /**
     * @brief Constructs the debug memory resource.
     *
     * @param[in] upstream The upstream memory resource to use.
     */
    VECMEM_CORE_EXPORT
    debug_memory_resource(memory_resource& upstream);
    /// Move constructor
    VECMEM_CORE_EXPORT
    debug_memory_resource(debug_memory_resource&& parent);
    /// Disallow copying the memory resource
    debug_memory_resource(const debug_memory_resource&) = delete;

    /// Destructor
    VECMEM_CORE_EXPORT
    ~debug_memory_resource();

    /// Move assignment operator
    VECMEM_CORE_EXPORT
    debug_memory_resource& operator=(debug_memory_resource&& rhs);
    /// Disallow copying the memory resource
    debug_memory_resource& operator=(const debug_memory_resource&) = delete;

private:
    /// @name Function(s) implementing @c vecmem:::memory_resource
    /// @{

    /// Allocate memory with one of the underlying resources
    VECMEM_CORE_EXPORT
    virtual void* do_allocate(std::size_t, std::size_t) override final;
    /// De-allocate a previously allocated memory block
    VECMEM_CORE_EXPORT
    virtual void do_deallocate(void* p, std::size_t,
                               std::size_t) override final;

    /// @}

    /// The implementation of the debug memory resource.
    std::unique_ptr<details::debug_memory_resource_impl> m_impl;

};  // class debug_memory_resource

}  // namespace vecmem
