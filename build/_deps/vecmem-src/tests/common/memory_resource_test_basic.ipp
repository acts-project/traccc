/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

/// Perform some very basic tests that do not need host accessibility
TEST_P(memory_resource_test_basic, allocations) {

    vecmem::memory_resource* resource = GetParam();
    for (std::size_t size = 0; size < 100000; size += 1000) {
        void* ptr = resource->allocate(size);
        resource->deallocate(ptr, size);
    }
}
