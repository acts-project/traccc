/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Test include(s).
#include "../common/copy_tests.hpp"
#include "../common/soa_copy_tests.hpp"

// VecMem include(s).
#include "vecmem/memory/sycl/device_memory_resource.hpp"
#include "vecmem/memory/sycl/host_memory_resource.hpp"
#include "vecmem/memory/sycl/shared_memory_resource.hpp"
#include "vecmem/utils/copy.hpp"
#include "vecmem/utils/sycl/async_copy.hpp"
#include "vecmem/utils/sycl/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <tuple>

// Objects used in the test(s).
static vecmem::sycl::queue_wrapper sycl_queue;
static vecmem::sycl::host_memory_resource sycl_host_resource{sycl_queue};
static vecmem::memory_resource* sycl_host_resource_ptr = &sycl_host_resource;
static vecmem::sycl::device_memory_resource sycl_device_resource{sycl_queue};
static vecmem::memory_resource* sycl_device_resource_ptr =
    &sycl_device_resource;
static vecmem::sycl::shared_memory_resource sycl_shared_resource{sycl_queue};
static vecmem::memory_resource* sycl_shared_resource_ptr =
    &sycl_shared_resource;
static vecmem::copy sycl_host_copy;
static vecmem::copy* sycl_host_copy_ptr = &sycl_host_copy;
static vecmem::sycl::copy sycl_device_copy{sycl_queue};
static vecmem::copy* sycl_device_copy_ptr = &sycl_device_copy;
static vecmem::sycl::async_copy sycl_async_device_copy{sycl_queue};
static vecmem::copy* sycl_async_device_copy_ptr = &sycl_async_device_copy;

/// The configurations to run the tests with.
static const auto sycl_copy_configs = testing::Values(
    std::tie(sycl_device_copy_ptr, sycl_host_copy_ptr, sycl_device_resource_ptr,
             sycl_host_resource_ptr),
    std::tie(sycl_async_device_copy_ptr, sycl_host_copy_ptr,
             sycl_device_resource_ptr, sycl_host_resource_ptr),
    std::tie(sycl_device_copy_ptr, sycl_host_copy_ptr, sycl_shared_resource_ptr,
             sycl_host_resource_ptr),
    std::tie(sycl_async_device_copy_ptr, sycl_host_copy_ptr,
             sycl_shared_resource_ptr, sycl_host_resource_ptr),
    std::tie(sycl_device_copy_ptr, sycl_host_copy_ptr, sycl_shared_resource_ptr,
             sycl_shared_resource_ptr),
    std::tie(sycl_async_device_copy_ptr, sycl_host_copy_ptr,
             sycl_shared_resource_ptr, sycl_shared_resource_ptr),
    std::tie(sycl_device_copy_ptr, sycl_host_copy_ptr, sycl_device_resource_ptr,
             sycl_shared_resource_ptr),
    std::tie(sycl_async_device_copy_ptr, sycl_host_copy_ptr,
             sycl_device_resource_ptr, sycl_shared_resource_ptr));

// Instantiate the test suite(s).
INSTANTIATE_TEST_SUITE_P(sycl_copy_tests, copy_tests, sycl_copy_configs);
INSTANTIATE_TEST_SUITE_P(sycl_soa_copy_tests_simple, soa_copy_tests_simple,
                         sycl_copy_configs);
INSTANTIATE_TEST_SUITE_P(sycl_soa_copy_tests_jagged, soa_copy_tests_jagged,
                         sycl_copy_configs);
