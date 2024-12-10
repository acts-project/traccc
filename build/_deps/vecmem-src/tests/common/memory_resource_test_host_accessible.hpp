/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/memory_resource.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

/// Test case for host-accessible memory resources
///
/// Providing a slightly more elaborate test for memory resources that can be
/// read/written from host code.
///
class memory_resource_test_host_accessible
    : public testing::TestWithParam<vecmem::memory_resource*> {

protected:
    /// Function performing some basic tests using @c vecmem::vector
    template <typename T>
    void test_host_accessible_resource(vecmem::vector<T>& test_vector);

};  // class memory_resource_test_host_accessible

// Include the implementation.
#include "memory_resource_test_host_accessible.ipp"
