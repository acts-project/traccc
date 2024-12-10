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
#include <cstddef>
#include <memory>

namespace vecmem {

// Forward declaration(s).
namespace details {
class contiguous_memory_resource_impl;
}

/**
 * @brief Downstream allocator that ensures that allocations are contiguous.
 *
 * When programming for co-processors, it is often desriable to keep
 * allocations contiguous. This downstream allocator fills that need. When
 * configured with an upstream memory resource, it will start out by
 * allocating a single, large, chunk of memory from the upstream. Then, it
 * will hand out pointers along that memory in a contiguous fashion. This
 * allocator guarantees that each consecutive allocation will start right at
 * the end of the previous.
 *
 * @note The allocation size on the upstream allocator is also the maximum
 * amount of memory that can be allocated from the contiguous memory
 * resource.
 */
class contiguous_memory_resource final : public details::memory_resource_base {

public:
    /**
     * @brief Constructs the contiguous memory resource.
     *
     * @param[in] upstream The upstream memory resource to use.
     * @param[in] size The size of memory to allocate upstream.
     */
    VECMEM_CORE_EXPORT
    contiguous_memory_resource(memory_resource& upstream, std::size_t size);
    /// Move constructor
    VECMEM_CORE_EXPORT
    contiguous_memory_resource(contiguous_memory_resource&& parent);
    /// Disallow copying the memory resource
    contiguous_memory_resource(const contiguous_memory_resource&) = delete;
    /**
     * @brief Deconstruct the contiguous memory resource.
     *
     * This method deallocates the arena memory on the upstream allocator.
     */
    VECMEM_CORE_EXPORT
    ~contiguous_memory_resource();

    /// Move assignment operator
    VECMEM_CORE_EXPORT
    contiguous_memory_resource& operator=(contiguous_memory_resource&& rhs);
    /// Disallow copying the memory resource
    contiguous_memory_resource& operator=(const contiguous_memory_resource&) =
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

    /// The implementation of the contiguous memory resource.
    std::unique_ptr<details::contiguous_memory_resource_impl> m_impl;

};  // class contiguous_memory_resource

}  // namespace vecmem
