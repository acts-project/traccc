/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../common/memory_resource_name_gen.hpp"
#include "../common/memory_resource_test_basic.hpp"
#include "../common/memory_resource_test_host_accessible.hpp"
#include "vecmem/memory/hip/device_memory_resource.hpp"
#include "vecmem/memory/hip/host_memory_resource.hpp"
#include "vecmem/memory/hip/managed_memory_resource.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// Memory resources.
static vecmem::hip::device_memory_resource device_resource;
static vecmem::hip::host_memory_resource host_resource;
static vecmem::hip::managed_memory_resource managed_resource;

// Instantiate the allocation tests on all of the resources.
INSTANTIATE_TEST_SUITE_P(hip_memory_resource_tests_basic,
                         memory_resource_test_basic,
                         testing::Values(&device_resource, &host_resource,
                                         &managed_resource),
                         vecmem::testing::memory_resource_name_gen(
                             {{&device_resource, "device_resource"},
                              {&host_resource, "host_resource"},
                              {&managed_resource, "managed_resource"}}));

// Instantiate the full test suite on the host-accessible memory resources.
INSTANTIATE_TEST_SUITE_P(hip_host_accessible_memory_resource_tests,
                         memory_resource_test_host_accessible,
                         testing::Values(&host_resource, &managed_resource),
                         vecmem::testing::memory_resource_name_gen(
                             {{&host_resource, "host_resource"},
                              {&managed_resource, "managed_resource"}}));
