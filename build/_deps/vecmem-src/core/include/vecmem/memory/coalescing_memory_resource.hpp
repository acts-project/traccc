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
#include <vector>

namespace vecmem {

// Forward declaration(s).
namespace details {
class coalescing_memory_resource_impl;
}

/**
 * @brief This memory resource tries to allocate with several upstream resources
 * and returns the first succesful one.
 */
class coalescing_memory_resource final : public details::memory_resource_base {

public:
    /**
     * @brief Constructs the coalescing memory resource.
     *
     * @note The memory resources passed to this constructed are given in order
     * of decreasing priority. That is to say, the first one is tried first,
     * then the second, etc.
     *
     * @param[in] upstreams The upstream memory resources to use.
     */
    VECMEM_CORE_EXPORT
    coalescing_memory_resource(
        std::vector<std::reference_wrapper<memory_resource>>&& upstreams);
    /// Move constructor
    VECMEM_CORE_EXPORT
    coalescing_memory_resource(coalescing_memory_resource&& parent);
    /// Disallow copying the memory resource
    coalescing_memory_resource(const coalescing_memory_resource&) = delete;

    /// Destructor
    VECMEM_CORE_EXPORT
    ~coalescing_memory_resource();

    /// Move assignment operator
    VECMEM_CORE_EXPORT
    coalescing_memory_resource& operator=(coalescing_memory_resource&& rhs);
    /// Disallow copying the memory resource
    coalescing_memory_resource& operator=(const coalescing_memory_resource&) =
        delete;

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

    /// The implementation of the coalescing memory resource.
    std::unique_ptr<details::coalescing_memory_resource_impl> m_impl;

};  // class coalescing_memory_resource

}  // namespace vecmem
