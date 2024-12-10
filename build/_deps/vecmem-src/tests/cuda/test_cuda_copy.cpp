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
#include "vecmem/memory/cuda/device_memory_resource.hpp"
#include "vecmem/memory/cuda/host_memory_resource.hpp"
#include "vecmem/memory/cuda/managed_memory_resource.hpp"
#include "vecmem/utils/copy.hpp"
#include "vecmem/utils/cuda/async_copy.hpp"
#include "vecmem/utils/cuda/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <memory>
#include <tuple>

// Objects used in the test(s).
static vecmem::cuda::host_memory_resource cuda_host_resource;
static vecmem::memory_resource* cuda_host_resource_ptr = &cuda_host_resource;
static vecmem::cuda::device_memory_resource cuda_device_resource;
static vecmem::memory_resource* cuda_device_resource_ptr =
    &cuda_device_resource;
static vecmem::cuda::managed_memory_resource cuda_managed_resource;
static vecmem::memory_resource* cuda_managed_resource_ptr =
    &cuda_managed_resource;
static vecmem::copy cuda_host_copy;
static vecmem::copy* cuda_host_copy_ptr = &cuda_host_copy;
static vecmem::cuda::copy cuda_device_copy;
static vecmem::copy* cuda_device_copy_ptr = &cuda_device_copy;
static std::unique_ptr<vecmem::cuda::stream_wrapper> cuda_stream;
static std::unique_ptr<vecmem::cuda::async_copy> cuda_async_device_copy;
static vecmem::copy* cuda_async_device_copy_ptr = nullptr;

/// Environment taking care of setting up and tearing down the async test types
class AsyncCUDACopyEnvironment : public ::testing::Environment {
public:
    /// Set up the environment
    void SetUp() override {
        cuda_stream = std::make_unique<vecmem::cuda::stream_wrapper>();
        cuda_async_device_copy =
            std::make_unique<vecmem::cuda::async_copy>(*cuda_stream);
        cuda_async_device_copy_ptr = cuda_async_device_copy.get();
    }
    /// Tear down the environment
    void TearDown() override {
        cuda_async_device_copy.reset();
        cuda_stream.reset();
        cuda_async_device_copy_ptr = nullptr;
    }
};

/// Register the environment
static ::testing::Environment* const async_cuda_copy_env =
    ::testing::AddGlobalTestEnvironment(new AsyncCUDACopyEnvironment{});

/// The configurations to run the tests with. Skip tests with asynchronous
/// copies on "managed host memory" on Windows. (Using managed memory as
/// "device memory" does seem to be fine.) There seems to be some issue with
/// CUDA, which results in SEH exceptions when scheduling an asynchronous
/// copy in such a setup, while a previous asynchronous copy is still running.
static const auto cuda_copy_configs = testing::Values(
    std::tie(cuda_device_copy_ptr, cuda_host_copy_ptr, cuda_device_resource_ptr,
             cuda_host_resource_ptr),
    std::tie(cuda_async_device_copy_ptr, cuda_host_copy_ptr,
             cuda_device_resource_ptr, cuda_host_resource_ptr),
    std::tie(cuda_device_copy_ptr, cuda_host_copy_ptr,
             cuda_managed_resource_ptr, cuda_host_resource_ptr),
    std::tie(cuda_async_device_copy_ptr, cuda_host_copy_ptr,
             cuda_managed_resource_ptr, cuda_host_resource_ptr),
    std::tie(cuda_device_copy_ptr, cuda_host_copy_ptr,
             cuda_managed_resource_ptr, cuda_managed_resource_ptr),
#ifndef _WIN32
    std::tie(cuda_async_device_copy_ptr, cuda_host_copy_ptr,
             cuda_managed_resource_ptr, cuda_managed_resource_ptr),
#endif
    std::tie(cuda_device_copy_ptr, cuda_host_copy_ptr, cuda_device_resource_ptr,
             cuda_managed_resource_ptr)
#ifndef _WIN32
        ,
    std::tie(cuda_async_device_copy_ptr, cuda_host_copy_ptr,
             cuda_device_resource_ptr, cuda_managed_resource_ptr)
#endif
);

// Instantiate the test suite(s).
INSTANTIATE_TEST_SUITE_P(cuda_copy_tests, copy_tests, cuda_copy_configs);
INSTANTIATE_TEST_SUITE_P(cuda_soa_copy_tests_simple, soa_copy_tests_simple,
                         cuda_copy_configs);
INSTANTIATE_TEST_SUITE_P(cuda_soa_copy_tests_jagged, soa_copy_tests_jagged,
                         cuda_copy_configs);
