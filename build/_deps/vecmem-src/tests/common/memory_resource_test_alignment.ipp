/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#include "vecmem/memory/details/is_aligned.hpp"

// System include(s).
#include <cmath>

/// Basic aligned memory allocation tests
TEST_P(memory_resource_test_alignment, basics) {

    // Get the memory resource to be used.
    vecmem::memory_resource* resource = GetParam();

    // Perform alignments for a set of different sizes.
    for (std::size_t size = 1000; size < 100000; size += 1000) {
        // And for a set of different alignments.
        static constexpr std::size_t max_alignment = 0x1 << 16;
        for (std::size_t alignment = 1; alignment <= max_alignment;
             alignment *= 2) {
            // Perform the allocation.
            void* ptr = resource->allocate(size, alignment);
            // Check that it succeeded.
            EXPECT_NE(ptr, nullptr);
            EXPECT_TRUE(vecmem::details::is_aligned(ptr, alignment));
            // Free the memory.
            resource->deallocate(ptr, size, alignment);
        }
    }
}
