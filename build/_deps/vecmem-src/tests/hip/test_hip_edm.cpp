/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../common/soa_device_tests.hpp"
#include "test_hip_edm_kernels.hpp"

// Project include(s).
#include "vecmem/memory/hip/device_memory_resource.hpp"
#include "vecmem/memory/hip/host_memory_resource.hpp"
#include "vecmem/utils/hip/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

/// Host memory resource to use in the tests.
static vecmem::hip::host_memory_resource hip_host_mr;
/// Device memory resource to use in the tests.
static vecmem::hip::device_memory_resource hip_device_mr;

/// Synchronous device copy object to use in the tests.
static vecmem::hip::copy hip_copy;

/// Pointer to the function filling a simple SoA container on a HIP device.
static void* hipSimpleFillPtr = reinterpret_cast<void*>(&hipSimpleFill);
/// Pointer to the function modifying a simple SoA container on a HIP device.
static void* hipSimpleModifyPtr = reinterpret_cast<void*>(&hipSimpleModify);

/// Pointer to the function filling a jagged SoA container on a HIP device.
static void* hipJaggedFillPtr = reinterpret_cast<void*>(&hipJaggedFill);
/// Pointer to the function modifying a jagged SoA container on a HIP device.
static void* hipJaggedModifyPtr = reinterpret_cast<void*>(&hipJaggedModify);

// Instantiate the test suites.
INSTANTIATE_TEST_SUITE_P(
    hip_soa_device_tests_simple, soa_device_tests_simple,
    testing::Values(std::tie(hip_host_mr, hip_device_mr, hip_host_mr, hip_copy,
                             hipSimpleFillPtr, hipSimpleModifyPtr)));
INSTANTIATE_TEST_SUITE_P(
    hip_soa_device_tests_jagged, soa_device_tests_jagged,
    testing::Values(std::tie(hip_host_mr, hip_device_mr, hip_host_mr, hip_copy,
                             hipJaggedFillPtr, hipJaggedModifyPtr)));
