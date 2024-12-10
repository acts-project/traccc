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
#include <functional>
#include <memory>

namespace vecmem {

// Forward declaration(s).
namespace details {
class choice_memory_resource_impl;
}

/**
 * @brief This memory resource conditionally allocates memory. It is
 * constructed with a function that determines which upstream resource to use.
 *
 * This resource can be used to construct complex conditional allocation
 * schemes.
 */
class choice_memory_resource final : public details::memory_resource_base {

public:
    /**
     * @brief Construct the choice memory resource.
     *
     * @param[in] upstreams The upstream memory resources to use.
     * @param[in] decision The function which picks the upstream memory
     * resource to use by index.
     */
    VECMEM_CORE_EXPORT
    choice_memory_resource(
        std::function<memory_resource&(std::size_t, std::size_t)> decision);
    /// Move constructor
    VECMEM_CORE_EXPORT
    choice_memory_resource(choice_memory_resource&& parent);
    /// Disallow copying the memory resource
    choice_memory_resource(const choice_memory_resource&) = delete;

    /// Destructor
    VECMEM_CORE_EXPORT
    ~choice_memory_resource();

    /// Move assignment operator
    VECMEM_CORE_EXPORT
    choice_memory_resource& operator=(choice_memory_resource&& rhs);
    /// Disallow copying the memory resource
    choice_memory_resource& operator=(const choice_memory_resource&) = delete;

private:
    /// @name Function(s) implementing @c vecmem::memory_resource
    /// @{

    /// Allocate memory with one of the underlying resources
    VECMEM_CORE_EXPORT
    virtual void* do_allocate(std::size_t, std::size_t) override final;
    /// De-allocate a previously allocated memory block
    VECMEM_CORE_EXPORT
    virtual void do_deallocate(void* p, std::size_t,
                               std::size_t) override final;

    /// @}

    /// The implementation of the choice memory resource.
    std::unique_ptr<details::choice_memory_resource_impl> m_impl;

};  // class choice_memory_resource

}  // namespace vecmem
