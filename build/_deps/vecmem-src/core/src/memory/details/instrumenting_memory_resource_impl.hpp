/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/instrumenting_memory_resource.hpp"
#include "vecmem/memory/memory_resource.hpp"

// System include(s).
#include <cstddef>
#include <functional>
#include <vector>

namespace vecmem::details {

/// Implementation for @c vecmem::details::instrumenting_memory_resource
class instrumenting_memory_resource_impl {

public:
    /**
     * @brief Constructs the instrumenting memory resource implementation.
     *
     * @param[in] upstream The upstream memory resource to use.
     */
    instrumenting_memory_resource_impl(memory_resource& upstream);

    /**
     * @brief Return a list of memory allocation and deallocation events in
     * chronological order.
     */
    const std::vector<instrumenting_memory_resource::memory_event>& get_events(
        void) const;

    /**
     * @brief Add a pre-allocation hook.
     *
     * Whenever memory is allocated, all pre-allocation hooks are exectuted.
     * This happens before we know whether the allocation was a success or not.
     *
     * The function passed to this function should accept the size of the
     * request as the first argument, and the alignment as the second.
     */
    void add_pre_allocate_hook(std::function<void(std::size_t, std::size_t)> f);

    /**
     * @brief Add a post-allocation hook.
     *
     * Whenever memory is allocated, all post-allocation hooks are exectuted.
     * This happens after we know whether the allocation was a success or not,
     * and the pointer that was returned.
     *
     * The function passed to this function should accept the size of the
     * request as the first argument, the alignment as the second, and the
     * pointer to the allocated memory as the third argument.
     */
    void add_post_allocate_hook(
        std::function<void(std::size_t, std::size_t, void*)> f);

    /**
     * @brief Add a pre-deallocation hook.
     *
     * Whenever memory is deallocated, all pre-deallocation hooks are
     * exectuted.]
     *
     * The function passed to this function should accept the pointer to
     * allocate as its first argument, the size of the request as the second
     * argument, and the alignment as the third.
     */
    void add_pre_deallocate_hook(
        std::function<void(void*, std::size_t, std::size_t)> f);

    /// Allocate memory with a upstream memory resource
    void* allocate(std::size_t, std::size_t);

    /// Deallocate previously allocated memory
    void deallocate(void* p, std::size_t, std::size_t);

private:
    /*
     * The upstream memory resource to which requests for allocation and
     * deallocation will be forwarded.
     */
    memory_resource& m_upstream;

    /*
     * This list stores a chronological set of requests that were passed to
     * this memory resource.
     */
    std::vector<instrumenting_memory_resource::memory_event> m_events;

    /*
     * The list of all pre-allocation hooks.
     */
    std::vector<std::function<void(std::size_t, std::size_t)>>
        m_pre_allocate_hooks;

    /*
     * The list of all post-allocation hooks.
     */
    std::vector<std::function<void(std::size_t, std::size_t, void*)>>
        m_post_allocate_hooks;

    /*
     * The list of all pre-deallocation hooks.
     */
    std::vector<std::function<void(void*, std::size_t, std::size_t)>>
        m_pre_deallocate_hooks;

};  // class instrumenting_memory_resource_impl

}  // namespace vecmem::details
