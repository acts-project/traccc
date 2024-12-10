/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "instrumenting_memory_resource_impl.hpp"

// System include(s).
#include <chrono>

namespace vecmem::details {

instrumenting_memory_resource_impl::instrumenting_memory_resource_impl(
    memory_resource &upstream)
    : m_upstream(upstream) {}

const std::vector<instrumenting_memory_resource::memory_event>
    &instrumenting_memory_resource_impl::get_events(void) const {

    return m_events;
}

void instrumenting_memory_resource_impl::add_pre_allocate_hook(
    std::function<void(std::size_t, std::size_t)> f) {

    m_pre_allocate_hooks.push_back(f);
}

void instrumenting_memory_resource_impl::add_post_allocate_hook(
    std::function<void(std::size_t, std::size_t, void *)> f) {

    m_post_allocate_hooks.push_back(f);
}

void instrumenting_memory_resource_impl::add_pre_deallocate_hook(
    std::function<void(void *, std::size_t, std::size_t)> f) {

    m_pre_deallocate_hooks.push_back(f);
}

void *instrumenting_memory_resource_impl::allocate(std::size_t size,
                                                   std::size_t align) {

    if (size == 0) {
        return nullptr;
    }

    /*
     * First, we will execute all pre-allocation hooks.
     */
    for (const std::function<void(std::size_t, std::size_t)> &f :
         m_pre_allocate_hooks) {
        f(size, align);
    }

    /*
     * We record the time before the request, so we can compute the total
     * execution time afterwards.
     */
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();

    void *ptr;

    /*
     * If an allocation fails, a std::bad_alloc exception is thrown. Normally
     * we can just forward those to the user, but in this case we want to do
     * some extra administration. Therefore, if this happens, we set the
     * pointer to null for now.
     */
    try {
        ptr = m_upstream.allocate(size, align);
    } catch (std::bad_alloc &) {
        ptr = nullptr;
    }

    /*
     * Record the time after the allocation, and compute the difference in
     * nanoseconds from the start of the allocation.
     */
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();

    std::size_t time = static_cast<std::size_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());

    /*
     * Add a new allocation event with the size, alignment, pointer, and time
     * of what has just happened.
     */
    m_events.emplace_back(
        instrumenting_memory_resource::memory_event::type::ALLOCATION, size,
        align, ptr, time);

    /*
     * Now, we can run the post-allocation hooks. For failed allocations, the
     * pointer will be null.
     */
    for (const std::function<void(std::size_t, std::size_t, void *)> &f :
         m_post_allocate_hooks) {
        f(size, align, ptr);
    }

    /*
     * Now we check whether our allocation failed. If that is the case, we just
     * throw another bad allocation exception.
     */
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }

    return ptr;
}

void instrumenting_memory_resource_impl::deallocate(void *ptr, std::size_t size,
                                                    std::size_t align) {

    if (ptr == nullptr) {
        return;
    }

    /*
     * First, we will run all of our pre-deallocation hooks.
     */
    for (const std::function<void(void *, std::size_t, std::size_t)> &f :
         m_pre_deallocate_hooks) {
        f(ptr, size, align);
    }

    /*
     * As with allocation, we calculate the time taken to process this
     * deallocation.
     */
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();

    /*
     * The deallocation, like allocation, is a forwarding method.
     */
    m_upstream.deallocate(ptr, size, align);

    /*
     * Compute the total elapsed time during the deallocation request.
     */
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();

    std::size_t time = static_cast<std::size_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());

    /*
     * Register a deallocation event.
     */
    m_events.emplace_back(
        instrumenting_memory_resource::memory_event::type::DEALLOCATION, size,
        align, ptr, time);
}

}  // namespace vecmem::details
