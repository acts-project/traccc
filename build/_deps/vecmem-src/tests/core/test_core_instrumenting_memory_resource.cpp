/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include <functional>

#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/memory/instrumenting_memory_resource.hpp"
#include "vecmem/utils/memory_monitor.hpp"

class core_instrumenting_memory_resource_test : public testing::Test {
protected:
    vecmem::host_memory_resource m_upstream;
};

TEST_F(core_instrumenting_memory_resource_test, equality) {
    vecmem::instrumenting_memory_resource res1(m_upstream);
    vecmem::instrumenting_memory_resource res2(m_upstream);

    EXPECT_FALSE(res1.is_equal(res2));
    EXPECT_TRUE(res1.is_equal(res1));
    EXPECT_TRUE(res2.is_equal(res2));
}

TEST_F(core_instrumenting_memory_resource_test, pre_allocate_hook) {
    vecmem::instrumenting_memory_resource res(m_upstream);

    std::size_t total_size = 0;

    res.add_pre_allocate_hook(
        [&total_size](std::size_t size, std::size_t) { total_size += size; });

    void* ptr1 = res.allocate(100);
    void* ptr2 = res.allocate(50);
    void* ptr3 = res.allocate(2);

    EXPECT_EQ(total_size, 152);

    res.deallocate(ptr1, 100);
    res.deallocate(ptr2, 50);
    res.deallocate(ptr3, 2);
}

TEST_F(core_instrumenting_memory_resource_test, post_allocate_hook) {
    vecmem::instrumenting_memory_resource res(m_upstream);

    std::size_t total_size = 0;

    res.add_post_allocate_hook([&total_size](std::size_t size, std::size_t,
                                             void*) { total_size += size; });

    void* ptr1 = res.allocate(100);
    void* ptr2 = res.allocate(50);
    void* ptr3 = res.allocate(2);

    EXPECT_EQ(total_size, 152);

    res.deallocate(ptr1, 100);
    res.deallocate(ptr2, 50);
    res.deallocate(ptr3, 2);
}

TEST_F(core_instrumenting_memory_resource_test, post_allocate_hook_eq) {
    vecmem::instrumenting_memory_resource res(m_upstream);

    void* last;

    res.add_post_allocate_hook(
        [&last](std::size_t, std::size_t, void* ptr) { last = ptr; });

    void* ptr1 = res.allocate(100);

    EXPECT_EQ(ptr1, last);

    void* ptr2 = res.allocate(50);

    EXPECT_EQ(ptr2, last);

    void* ptr3 = res.allocate(2);

    EXPECT_EQ(ptr3, last);

    res.deallocate(ptr1, 100);
    res.deallocate(ptr2, 50);
    res.deallocate(ptr3, 2);
}

TEST_F(core_instrumenting_memory_resource_test, pre_post_allocate_hook) {
    vecmem::instrumenting_memory_resource res(m_upstream);

    std::size_t total_size = 0;

    res.add_pre_allocate_hook(
        [&total_size](std::size_t size, std::size_t) { total_size += size; });

    res.add_post_allocate_hook([&total_size](std::size_t size, std::size_t,
                                             void*) { total_size += size; });

    void* ptr1 = res.allocate(100);
    void* ptr2 = res.allocate(50);
    void* ptr3 = res.allocate(2);

    EXPECT_EQ(total_size, 304);

    res.deallocate(ptr1, 100);
    res.deallocate(ptr2, 50);
    res.deallocate(ptr3, 2);
}

TEST_F(core_instrumenting_memory_resource_test, pre_deallocate_hook) {
    vecmem::instrumenting_memory_resource res(m_upstream);

    std::size_t total_size = 0;

    res.add_pre_deallocate_hook(
        [&total_size](void*, std::size_t size, std::size_t) {
            total_size += size;
        });

    void* ptr1 = res.allocate(100);
    void* ptr2 = res.allocate(50);
    void* ptr3 = res.allocate(2);

    res.deallocate(ptr1, 100);
    res.deallocate(ptr2, 50);

    EXPECT_EQ(total_size, 150);

    res.deallocate(ptr3, 2);

    EXPECT_EQ(total_size, 152);
}

TEST_F(core_instrumenting_memory_resource_test, events) {
    vecmem::instrumenting_memory_resource res(m_upstream);

    void* ptr1 = res.allocate(100);
    void* ptr2 = res.allocate(50);
    res.deallocate(ptr2, 50);
    void* ptr3 = res.allocate(2, 8);
    res.deallocate(ptr1, 100);

    const std::vector<vecmem::instrumenting_memory_resource::memory_event>
        events = res.get_events();

    ASSERT_EQ(events.size(), 5);

    EXPECT_EQ(
        events[0].m_type,
        vecmem::instrumenting_memory_resource::memory_event::type::ALLOCATION);
    EXPECT_EQ(events[0].m_size, 100);
    EXPECT_EQ(events[0].m_ptr, ptr1);

    EXPECT_EQ(
        events[1].m_type,
        vecmem::instrumenting_memory_resource::memory_event::type::ALLOCATION);
    EXPECT_EQ(events[1].m_size, 50);
    EXPECT_EQ(events[1].m_ptr, ptr2);

    EXPECT_EQ(events[2].m_type, vecmem::instrumenting_memory_resource::
                                    memory_event::type::DEALLOCATION);
    EXPECT_EQ(events[2].m_size, 50);
    EXPECT_EQ(events[2].m_ptr, ptr2);

    EXPECT_EQ(
        events[3].m_type,
        vecmem::instrumenting_memory_resource::memory_event::type::ALLOCATION);
    EXPECT_EQ(events[3].m_size, 2);
    EXPECT_EQ(events[3].m_align, 8);
    EXPECT_EQ(events[3].m_ptr, ptr3);

    EXPECT_EQ(events[4].m_type, vecmem::instrumenting_memory_resource::
                                    memory_event::type::DEALLOCATION);
    EXPECT_EQ(events[4].m_size, 100);
    EXPECT_EQ(events[4].m_ptr, ptr1);

    res.deallocate(ptr3, 2, 8);
}

TEST_F(core_instrumenting_memory_resource_test, memory_monitor) {

    // Set up the memory resource
    vecmem::instrumenting_memory_resource res(m_upstream);

    // Set up the memory monitor
    vecmem::memory_monitor monitor(res);

    // Perform some allocations and de-allocations
    void* ptr1 = res.allocate(200);
    void* ptr2 = res.allocate(500);
    res.deallocate(ptr1, 200);
    ptr1 = res.allocate(600);
    void* ptr3 = res.allocate(100);
    res.deallocate(ptr1, 600);
    res.deallocate(ptr2, 500);

    // Check the statistics.
    EXPECT_EQ(monitor.total_allocation(), 1400);
    EXPECT_EQ(monitor.outstanding_allocation(), 100);
    EXPECT_EQ(monitor.average_allocation(), 350);
    EXPECT_EQ(monitor.maximal_allocation(), 1200);

    // Clean up.
    res.deallocate(ptr3, 100);
    EXPECT_EQ(monitor.outstanding_allocation(), 0);
}
