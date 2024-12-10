/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include "vecmem/memory/conditional_memory_resource.hpp"
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/memory/instrumenting_memory_resource.hpp"

TEST(core_conditional_memory_resource_test, allocate1) {
    vecmem::host_memory_resource ups;
    vecmem::instrumenting_memory_resource mon(ups);
    vecmem::conditional_memory_resource res(
        mon, [](std::size_t s, std::size_t) { return s >= 1024; });

    std::size_t allocs = 0;

    mon.add_pre_allocate_hook(
        [&allocs](std::size_t, std::size_t) { ++allocs; });

    void* p = nullptr;

    EXPECT_THROW(p = res.allocate(10), std::bad_alloc);
    EXPECT_NO_THROW(p = res.allocate(0));
    EXPECT_EQ(allocs, 0);
    EXPECT_NO_THROW(p = res.allocate(5321));
    EXPECT_EQ(allocs, 1);
    res.deallocate(p, 5321);
    EXPECT_THROW(p = res.allocate(55), std::bad_alloc);
    EXPECT_THROW(p = res.allocate(19), std::bad_alloc);
    EXPECT_THROW(p = res.allocate(10, 16), std::bad_alloc);
    EXPECT_THROW(p = res.allocate(10, 4), std::bad_alloc);
    EXPECT_THROW(p = res.allocate(10, 32), std::bad_alloc);
    EXPECT_EQ(allocs, 1);
    EXPECT_NO_THROW(p = res.allocate(1024));
    res.deallocate(p, 1024);
    EXPECT_EQ(allocs, 2);
    EXPECT_NO_THROW(p = res.allocate(5000));
    res.deallocate(p, 5000);
    EXPECT_EQ(allocs, 3);
}

TEST(core_conditional_memory_resource_test, allocate2) {
    vecmem::host_memory_resource ups;
    vecmem::instrumenting_memory_resource mon(ups);
    vecmem::conditional_memory_resource res(
        mon, [](std::size_t, std::size_t a) { return a >= 32; });

    std::size_t allocs = 0;

    mon.add_pre_allocate_hook(
        [&allocs](std::size_t, std::size_t) { ++allocs; });

    void* p = nullptr;

    EXPECT_THROW(p = res.allocate(10), std::bad_alloc);
    EXPECT_NO_THROW(p = res.allocate(0));
    EXPECT_THROW(p = res.allocate(5321), std::bad_alloc);
    EXPECT_THROW(p = res.allocate(55), std::bad_alloc);
    EXPECT_THROW(p = res.allocate(19), std::bad_alloc);
    EXPECT_THROW(p = res.allocate(10, 16), std::bad_alloc);
    EXPECT_THROW(p = res.allocate(10, 4), std::bad_alloc);
    EXPECT_EQ(allocs, 0);
    EXPECT_NO_THROW(p = res.allocate(10, 32));
    EXPECT_EQ(allocs, 1);
    res.deallocate(p, 10, 32);
    EXPECT_THROW(p = res.allocate(1024), std::bad_alloc);
    EXPECT_THROW(p = res.allocate(5000), std::bad_alloc);
    EXPECT_EQ(allocs, 1);
    EXPECT_NO_THROW(p = res.allocate(1024, 128));
    EXPECT_EQ(allocs, 2);
    res.deallocate(p, 1024, 128);
    EXPECT_NO_THROW(p = res.allocate(5000, 256));
    res.deallocate(p, 5000, 256);
    EXPECT_EQ(allocs, 3);
}
