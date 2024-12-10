/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
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
class pool_memory_resource_impl;
}

/// Memory resource pooling allocations of various sizes
///
/// This is a "downstream" memory resource allowing for pooling and caching
/// allocations from an upstream resource.
///
/// The code is a copy of @c thrust::mr::disjoint_unsynchronized_pool_resource,
/// giving it a standard @c std::pmr::memory_resource interface. (And
/// simplifying it in some places a little.)
///
/// The original code of @c thrust::mr::disjoint_unsynchronized_pool_resource
/// is licensed under the Apache License, Version 2.0, which is available at:
/// http://www.apache.org/licenses/LICENSE-2.0
///
class pool_memory_resource final : public details::memory_resource_base {

public:
    /// Runtime options for @c vecmem::pool_memory_resource
    struct VECMEM_CORE_EXPORT options {

        /// Default constructor
        ///
        /// It is necessary to work around issue:
        /// https://github.com/llvm/llvm-project/issues/36032
        ///
        options();

        /// The minimal number of blocks, i.e. pieces of memory handed off to
        /// the user from a pool of a given size, in a single chunk allocated
        /// from upstream.
        std::size_t min_blocks_per_chunk = 16;
        /// The minimal number of bytes in a single chunk allocated from
        /// upstream.
        std::size_t min_bytes_per_chunk = 1024;
        /// The maximal number of blocks, i.e. pieces of memory handed off to
        /// the user from a pool of a given size, in a single chunk allocated
        /// from upstream.
        std::size_t max_blocks_per_chunk = static_cast<std::size_t>(1) << 20;
        /// The maximal number of bytes in a single chunk allocated from
        /// upstream.
        std::size_t max_bytes_per_chunk = static_cast<std::size_t>(1) << 30;

        /// The size of blocks in the smallest pool covered by the pool
        /// resource. All allocation requests below this size will be rounded up
        /// to this size.
        std::size_t smallest_block_size = alignof(std::max_align_t);
        /// The size of blocks in the largest pool covered by the pool resource.
        /// All allocation requests above this size will be considered
        /// oversized, allocated directly from upstream (and not from a pool),
        /// and cached only if @c cache_oversized is @c true.
        std::size_t largest_block_size = static_cast<std::size_t>(1) << 20;

        /// The alignment of all blocks in internal pools of the pool resource.
        /// All allocation requests above this alignment will be considered
        /// oversized, allocated directly from upstream (and not from a pool),
        /// and cached only if @c cache_oversized is @c true.
        std::size_t alignment = alignof(std::max_align_t);

        /// Decides whether oversized and overaligned blocks are cached for
        /// later use, or immediately return it to the upstream resource.
        bool cache_oversized = true;

        /// The size factor at which a cached allocation is considered too
        /// ridiculously oversized to use to fulfill an allocation request. For
        /// instance: the user requests an allocation of size 1024 bytes. A
        /// block of size 32 * 1024 bytes is cached. If @c
        /// cached_size_cutoff_factor is 32 or less, this block will be
        /// considered too big for that allocation request.
        std::size_t cached_size_cutoff_factor = 16;
        /// The alignment factor at which a cached allocation is considered too
        /// ridiculously overaligned to use to fulfill an allocation request.
        /// For instance: the user requests an allocation aligned to 32 bytes. A
        /// block aligned to 1024 bytes is cached. If @c
        /// cached_size_cutoff_factor is 32 or less, this block will be
        /// considered too overaligned for that allocation request.
        std::size_t cached_alignment_cutoff_factor = 16;

    };  // struct options

    /// Create a pool memory resource with the given options
    ///
    /// @param upstream The upstream memory resource to use for allocations
    /// @param opts The options to use for the pool memory resource
    ///
    VECMEM_CORE_EXPORT
    pool_memory_resource(memory_resource& upstream,
                         const options& opts = options{});
    /// Move constructor
    VECMEM_CORE_EXPORT
    pool_memory_resource(pool_memory_resource&& parent);
    /// Disallow copying the memory resource
    pool_memory_resource(const pool_memory_resource&) = delete;

    /// Destructor, freeing all allocations
    VECMEM_CORE_EXPORT
    ~pool_memory_resource();

    /// Move assignment operator
    VECMEM_CORE_EXPORT
    pool_memory_resource& operator=(pool_memory_resource&& rhs);
    /// Disallow copying the memory resource
    pool_memory_resource& operator=(const pool_memory_resource&) = delete;

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
    std::unique_ptr<details::pool_memory_resource_impl> m_impl;

};  // class pool_memory_resource

}  // namespace vecmem
