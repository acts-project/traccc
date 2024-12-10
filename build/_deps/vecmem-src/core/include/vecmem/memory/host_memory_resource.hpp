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
#include "vecmem/vecmem_core_export.hpp"

namespace vecmem {

/**
 * @brief Memory resource which wraps standard library memory allocation calls.
 *
 * This is probably the simplest memory resource you can possibly write. It
 * is a terminal resource which does nothing but wrap @c std::aligned_alloc and
 * @c std::free. It is state-free (on the relevant levels of abstraction).
 */
class host_memory_resource final : public details::memory_resource_base {

public:
    /// Default constructor
    VECMEM_CORE_EXPORT host_memory_resource();
    /// Destructor
    VECMEM_CORE_EXPORT ~host_memory_resource();

protected:
    /// @name Function(s) implementing @c vecmem::memory_resource
    /// @{

    /// Allocate standard host memory
    VECMEM_CORE_EXPORT
    virtual void* do_allocate(std::size_t size,
                              std::size_t alignment) override final;
    /// De-allocate a block of previously allocated memory
    VECMEM_CORE_EXPORT
    virtual void do_deallocate(void* p, std::size_t size,
                               std::size_t alignment) override final;
    /// Compares @c *this for equality with @c other
    VECMEM_CORE_EXPORT
    virtual bool do_is_equal(
        const memory_resource& other) const noexcept override final;

    /// @}

};  // class host_memory_resource

}  // namespace vecmem
