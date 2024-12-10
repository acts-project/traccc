/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/memory/pool_memory_resource.hpp"

// System include(s).
#include <functional>
#include <vector>

namespace vecmem::details {

/// Implementation of @c vecmem::pool_memory_resource
class pool_memory_resource_impl {

public:
    /// Constructor, on top of another memory resource
    pool_memory_resource_impl(memory_resource& upstream,
                              const pool_memory_resource::options& opts);

    /// Destructor, freeing all allocations
    ~pool_memory_resource_impl();

    /// Allocate memory
    void* allocate(std::size_t bytes, std::size_t alignment);

    /// Deallocate memory
    void deallocate(void* ptr, std::size_t bytes, std::size_t alignment);

private:
    /// The upstream memory resource
    std::reference_wrapper<memory_resource> m_upstream;
    /// The options for the pool memory resource
    pool_memory_resource::options m_options;

    /// Descriptor for a single chunk of allocated memory
    struct chunk_descriptor {
        /// Size of the memory chunk
        std::size_t size = 0u;
        /// Pointer to the beginning of the memory chunk
        void* pointer = nullptr;
    };

    /// Descriptor for a single, oversized block of memory
    struct oversized_block_descriptor {
        /// The size of the memory block
        std::size_t size = 0u;
        /// The alignment of the memory block
        std::size_t alignment = 0u;
        /// Pointer to the memory block
        void* pointer = nullptr;

        /// Comparison operator
        bool operator<(const oversized_block_descriptor& other) const;
    };

    /// Pool of available memory blocks of a given size/alignment
    struct pool {
        /// Available blocks, ready for use
        std::vector<void*> free_blocks;
        std::size_t previous_allocated_count = 0u;
    };

    /// Helper variable, with the base-2 log of the smallest block size
    const std::size_t m_smallest_block_log2;

    /// Buckets containing free lists for each pooled size
    std::vector<pool> m_pools;
    /// List of all allocations from the upstream memory resource
    std::vector<chunk_descriptor> m_allocated;
    /// List of all cached oversized/overaligned blocks that have been returned
    /// to the pool to cache
    std::vector<oversized_block_descriptor> m_cached_oversized;
    /// List of all oversized/overaligned allocations from the upstream memory
    /// resource
    std::vector<oversized_block_descriptor> m_oversized;

};  // class pool_memory_resource_impl

}  // namespace vecmem::details
