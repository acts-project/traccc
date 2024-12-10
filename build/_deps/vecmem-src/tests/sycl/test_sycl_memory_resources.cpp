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
#include "vecmem/memory/sycl/device_memory_resource.hpp"
#include "vecmem/memory/sycl/host_memory_resource.hpp"
#include "vecmem/memory/sycl/shared_memory_resource.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// Memory resources.
static vecmem::sycl::device_memory_resource device_resource;
static vecmem::sycl::host_memory_resource host_resource;
static vecmem::sycl::shared_memory_resource shared_resource;

// Instantiate the allocation tests on all of the resources.
INSTANTIATE_TEST_SUITE_P(sycl_memory_resource_tests, memory_resource_test_basic,
                         testing::Values(&device_resource, &host_resource,
                                         &shared_resource),
                         vecmem::testing::memory_resource_name_gen(
                             {{&device_resource, "device_resource"},
                              {&host_resource, "host_resource"},
                              {&shared_resource, "shared_resource"}}));

// Instantiate the full test suite on the host-accessible memory resources.
INSTANTIATE_TEST_SUITE_P(sycl_host_accessible_memory_resource_tests,
                         memory_resource_test_host_accessible,
                         testing::Values(&host_resource, &shared_resource),
                         vecmem::testing::memory_resource_name_gen(
                             {{&host_resource, "host_resource"},
                              {&shared_resource, "shared_resource"}}));
