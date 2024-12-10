/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include "vecmem/memory/choice_memory_resource.hpp"
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/memory/instrumenting_memory_resource.hpp"
#include "vecmem/memory/memory_resource.hpp"

TEST(core_choice_memory_resource_test, allocate) {
    vecmem::host_memory_resource ups;

    vecmem::instrumenting_memory_resource mon1(ups);
    vecmem::instrumenting_memory_resource mon2(ups);
    vecmem::instrumenting_memory_resource mon3(ups);

    vecmem::choice_memory_resource res(
        [&mon1, &mon2, &mon3](std::size_t s,
                              std::size_t a) -> vecmem::memory_resource& {
            if (s >= 1024) {
                if (a >= 128) {
                    return mon1;
                } else {
                    return mon2;
                }
            } else {
                if (a >= 128) {
                    return mon3;
                } else {
                    throw std::bad_alloc();
                }
            }
        });

    std::size_t last_alloc = 0;

    mon1.add_pre_allocate_hook(
        [&last_alloc](std::size_t, std::size_t) { last_alloc = 1; });

    mon2.add_pre_allocate_hook(
        [&last_alloc](std::size_t, std::size_t) { last_alloc = 2; });

    mon3.add_pre_allocate_hook(
        [&last_alloc](std::size_t, std::size_t) { last_alloc = 3; });

    void* p = nullptr;

    EXPECT_NO_THROW(p = res.allocate(2048, 32));
    res.deallocate(p, 2048, 32);
    EXPECT_EQ(last_alloc, 2);

    EXPECT_NO_THROW(p = res.allocate(512, 512));
    res.deallocate(p, 512, 512);
    EXPECT_EQ(last_alloc, 3);

    EXPECT_NO_THROW(p = res.allocate(4096, 256));
    res.deallocate(p, 4096, 256);
    EXPECT_EQ(last_alloc, 1);

    EXPECT_THROW(p = res.allocate(512, 32), std::bad_alloc);
}
