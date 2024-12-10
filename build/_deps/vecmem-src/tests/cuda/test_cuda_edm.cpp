/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../common/soa_device_tests.hpp"
#include "test_cuda_edm_kernels.hpp"

// Project include(s).
#include "vecmem/memory/cuda/device_memory_resource.hpp"
#include "vecmem/memory/cuda/host_memory_resource.hpp"
#include "vecmem/memory/cuda/managed_memory_resource.hpp"
#include "vecmem/utils/cuda/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

/// Host memory resource to use in the tests.
static vecmem::cuda::host_memory_resource cuda_host_mr;
/// Device memory resource to use in the tests.
static vecmem::cuda::device_memory_resource cuda_device_mr;
/// Managed memory resource to use in the tests.
static vecmem::cuda::managed_memory_resource cuda_managed_mr;

/// Synchronous device copy object to use in the tests.
static vecmem::cuda::copy cuda_copy;

/// Pointer to the function filling a simple SoA container on a CUDA device.
static void* cudaSimpleFillPtr = reinterpret_cast<void*>(&cudaSimpleFill);
/// Pointer to the function modifying a simple SoA container on a CUDA device.
static void* cudaSimpleModifyPtr = reinterpret_cast<void*>(&cudaSimpleModify);

/// Pointer to the function filling a jagged SoA container on a CUDA device.
static void* cudaJaggedFillPtr = reinterpret_cast<void*>(&cudaJaggedFill);
/// Pointer to the function modifying a jagged SoA container on a CUDA device.
static void* cudaJaggedModifyPtr = reinterpret_cast<void*>(&cudaJaggedModify);

// Instantiate the test suites.
INSTANTIATE_TEST_SUITE_P(cuda_soa_device_tests_simple, soa_device_tests_simple,
                         testing::Values(std::tie(cuda_host_mr, cuda_device_mr,
                                                  cuda_managed_mr, cuda_copy,
                                                  cudaSimpleFillPtr,
                                                  cudaSimpleModifyPtr)));
INSTANTIATE_TEST_SUITE_P(cuda_soa_device_tests_jagged, soa_device_tests_jagged,
                         testing::Values(std::tie(cuda_host_mr, cuda_device_mr,
                                                  cuda_managed_mr, cuda_copy,
                                                  cudaJaggedFillPtr,
                                                  cudaJaggedModifyPtr)));
