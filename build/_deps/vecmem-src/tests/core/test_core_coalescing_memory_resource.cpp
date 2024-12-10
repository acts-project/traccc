/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include "vecmem/memory/coalescing_memory_resource.hpp"
#include "vecmem/memory/conditional_memory_resource.hpp"
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/memory/instrumenting_memory_resource.hpp"
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/memory/terminal_memory_resource.hpp"

TEST(core_coalescing_memory_resource_test, allocate_terminal) {
    vecmem::terminal_memory_resource ter;
    vecmem::coalescing_memory_resource res({ter, ter, ter, ter});

    void *p = nullptr;

    EXPECT_THROW(p = res.allocate(1024), std::bad_alloc);

    static_cast<void>(p);
}

TEST(core_coalescing_memory_resource_test, allocate_basic) {
    vecmem::terminal_memory_resource ter;
    vecmem::host_memory_resource ups;
    vecmem::coalescing_memory_resource res({ter, ter, ter, ups, ter});

    void *p = nullptr;

    EXPECT_NO_THROW(p = res.allocate(1024));
    res.deallocate(p, 1024);
}

TEST(core_coalescing_memory_resource_test, allocate_complex) {
    vecmem::terminal_memory_resource ter;
    vecmem::host_memory_resource ups;

    vecmem::conditional_memory_resource con1(
        ups, [](std::size_t s, std::size_t) { return s >= 1024 && s < 8192; });
    vecmem::conditional_memory_resource con2(
        ups, [](std::size_t s, std::size_t) { return s >= 4096 && s < 65536; });

    vecmem::instrumenting_memory_resource mon1(con1);
    vecmem::instrumenting_memory_resource mon2(con2);

    std::size_t last_alloc = 0;
    std::size_t last_dealloc = 0;

    mon1.add_post_allocate_hook(
        [&last_alloc](std::size_t, std::size_t, void *p) {
            if (p != nullptr) {
                last_alloc = 1;
            }
        });

    mon1.add_pre_deallocate_hook(
        [&last_dealloc](void *, std::size_t, std::size_t) {
            last_dealloc = 1;
        });

    mon2.add_post_allocate_hook(
        [&last_alloc](std::size_t, std::size_t, void *p) {
            if (p != nullptr) {
                last_alloc = 2;
            }
        });

    mon2.add_pre_deallocate_hook(
        [&last_dealloc](void *, std::size_t, std::size_t) {
            last_dealloc = 2;
        });

    vecmem::coalescing_memory_resource res({ter, mon1, mon2, ter});

    void *p = nullptr;

    EXPECT_NO_THROW(p = res.allocate(1024));
    EXPECT_EQ(last_alloc, 1);
    res.deallocate(p, 1024);
    EXPECT_EQ(last_dealloc, 1);

    EXPECT_NO_THROW(p = res.allocate(4096));
    EXPECT_EQ(last_alloc, 1);
    res.deallocate(p, 4096);
    EXPECT_EQ(last_dealloc, 1);

    EXPECT_NO_THROW(p = res.allocate(8192));
    EXPECT_EQ(last_alloc, 2);
    res.deallocate(p, 1024);
    EXPECT_EQ(last_dealloc, 2);

    EXPECT_NO_THROW(p = res.allocate(4096));
    EXPECT_EQ(last_alloc, 1);
    res.deallocate(p, 4096);
    EXPECT_EQ(last_dealloc, 1);

    EXPECT_THROW(p = res.allocate(512), std::bad_alloc);

    EXPECT_THROW(p = res.allocate(131072), std::bad_alloc);
}
