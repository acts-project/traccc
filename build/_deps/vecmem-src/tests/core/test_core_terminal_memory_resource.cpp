/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/memory/terminal_memory_resource.hpp"

TEST(core_terminal_memory_resource_test, allocate) {
    vecmem::terminal_memory_resource res;

    void* p;

    EXPECT_THROW(p = res.allocate(10), std::bad_alloc);
    EXPECT_THROW(p = res.allocate(0), std::bad_alloc);
    EXPECT_THROW(p = res.allocate(5321532153), std::bad_alloc);
    EXPECT_THROW(p = res.allocate(55), std::bad_alloc);
    EXPECT_THROW(p = res.allocate(19), std::bad_alloc);
    EXPECT_THROW(p = res.allocate(10, 16), std::bad_alloc);
    EXPECT_THROW(p = res.allocate(10, 4), std::bad_alloc);
    EXPECT_THROW(p = res.allocate(10, 32), std::bad_alloc);

    static_cast<void>(p);
}

TEST(core_terminal_memory_resource_test, deallocate) {
    vecmem::host_memory_resource ups;
    vecmem::terminal_memory_resource res(ups);

    void* p = ups.allocate(1024);

    void* i = nullptr;

    /*
     * How do you even meaningfully test that a bit of code does nothing?
     */
    EXPECT_NO_THROW(res.deallocate(i, 8));
    EXPECT_NO_THROW(res.deallocate(p, 1024));

    ups.deallocate(p, 1024);
}
