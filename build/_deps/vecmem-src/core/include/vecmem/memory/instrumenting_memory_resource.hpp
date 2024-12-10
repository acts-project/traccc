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
#include <functional>
#include <memory>
#include <vector>

namespace vecmem {

// Forward declaration(s).
namespace details {
class instrumenting_memory_resource_impl;
}

/**
 * @brief This memory resource forwards allocation and deallocation requests to
 * the upstream resource while recording useful statistics and information
 * about these events.
 *
 * This allocator is here to allow us to debug, to profile, to test, but also
 * to instrument user code.
 */
class instrumenting_memory_resource final
    : public details::memory_resource_base {

public:
    /**
     * @brief Constructs the debug memory resource.
     *
     * @param[in] upstream The upstream memory resource to use.
     */
    VECMEM_CORE_EXPORT
    instrumenting_memory_resource(memory_resource& upstream);
    /// Move constructor
    VECMEM_CORE_EXPORT
    instrumenting_memory_resource(instrumenting_memory_resource&& parent);
    /// Disallow copying the memory resource
    instrumenting_memory_resource(const instrumenting_memory_resource&) =
        delete;

    /// Destructor
    VECMEM_CORE_EXPORT
    ~instrumenting_memory_resource();

    /// Move assignment operator
    VECMEM_CORE_EXPORT
    instrumenting_memory_resource& operator=(
        instrumenting_memory_resource&& rhs);
    /// Disallow copying the memory resource
    instrumenting_memory_resource& operator=(
        const instrumenting_memory_resource&) = delete;

    /**
     * @brief Structure describing a memory resource event.
     */
    struct VECMEM_CORE_EXPORT memory_event {

        /**
         * @brief Classify an event as an alloction or a deallocation.
         */
        enum class type { ALLOCATION, DEALLOCATION };

        /**
         * @brief Construct an allocation/deallocation event.
         *
         * @param[in] t The type of event (allocation or deallocation).
         * @param[in] s The size of the request.
         * @param[in] a The alignment of the request.
         * @param[in] p The pointer that was returned or deallocated.
         * @param[in] ns The time taken to perform the request in nanoseconds.
         */
        memory_event(type t, std::size_t s, std::size_t a, void* p,
                     std::size_t ns);

        /// The type of event (allocation or deallocation).
        type m_type;

        /// The size of the request.
        std::size_t m_size;
        /// The alignment of the request.
        std::size_t m_align;

        /// The pointer that was returned or deallocated.
        void* m_ptr;

        /// The time taken to perform the request in nanoseconds.
        std::size_t m_time;

    };  // struct memory_event

    /**
     * @brief Return a list of memory allocation and deallocation events in
     * chronological order.
     */
    VECMEM_CORE_EXPORT
    const std::vector<memory_event>& get_events(void) const;

    /**
     * @brief Add a pre-allocation hook.
     *
     * Whenever memory is allocated, all pre-allocation hooks are exectuted.
     * This happens before we know whether the allocation was a success or not.
     *
     * The function passed to this function should accept the size of the
     * request as the first argument, and the alignment as the second.
     */
    VECMEM_CORE_EXPORT
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
    VECMEM_CORE_EXPORT
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
    VECMEM_CORE_EXPORT
    void add_pre_deallocate_hook(
        std::function<void(void*, std::size_t, std::size_t)> f);

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

    /// The implementation of the debug memory resource.
    std::unique_ptr<details::instrumenting_memory_resource_impl> m_impl;

};  // class instrumenting_memory_resource

}  // namespace vecmem
