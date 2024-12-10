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

// System include(s).
#include <cstddef>

namespace vecmem {

/**
 * @brief This memory resource does nothing, but it does nothing for a purpose.
 *
 * This allocator has little practical use, but can be useful for defining some
 * conditional allocation schemes.
 *
 * Reimplementation of @c std::pmr::null_memory_resource but can accept another
 * memory resource in its constructor.
 */
class terminal_memory_resource final : public details::memory_resource_base {

public:
    /**
     * @brief Constructs the terminal memory resource, without an upstream
     * resource.
     */
    VECMEM_CORE_EXPORT
    terminal_memory_resource(void);
    /**
     * @brief Constructs the terminal memory resource, with an upstream
     * resource.
     *
     * @param[in] upstream The upstream memory resource to use.
     */
    VECMEM_CORE_EXPORT
    terminal_memory_resource(memory_resource& upstream);
    /// Destructor
    VECMEM_CORE_EXPORT
    ~terminal_memory_resource();

private:
    /// @name Function(s) implementing @c vecmem::memory_resource
    /// @{

    /// Throw @c std::bad_alloc.
    VECMEM_CORE_EXPORT
    virtual void* do_allocate(std::size_t, std::size_t) override final;
    /// Do nothing.
    VECMEM_CORE_EXPORT
    virtual void do_deallocate(void* p, std::size_t,
                               std::size_t) override final;
    /// Check whether the other resource is also a terminal resource.
    VECMEM_CORE_EXPORT
    virtual bool do_is_equal(
        const memory_resource& other) const noexcept override final;

    /// @}

};  // class terminal_memory_resource

}  // namespace vecmem
