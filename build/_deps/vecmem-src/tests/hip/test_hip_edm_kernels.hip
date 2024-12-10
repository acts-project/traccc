/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../common/jagged_soa_container_helpers.hpp"
#include "../common/simple_soa_container_helpers.hpp"
#include "test_hip_edm_kernels.hpp"

// Project include(s).
#include "../../hip/src/utils/hip_error_handling.hpp"

// HIP include(s).
#include <hip/hip_runtime.h>

__global__ void hipSimpleFillKernel(
    vecmem::testing::simple_soa_container::view view) {

    // Get the thread index.
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Run the modification using the helper function.
    vecmem::testing::simple_soa_container::device device{view};
    vecmem::testing::fill(i, device);
}

bool hipSimpleFill(vecmem::testing::simple_soa_container::view view) {

    // Launch the kernel.
    const unsigned int blockSize = 256;
    const unsigned int gridSize = (view.capacity() + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(hipSimpleFillKernel, gridSize, blockSize, 0, 0, view);

    // Check whether it succeeded to run.
    VECMEM_HIP_ERROR_CHECK(hipGetLastError());
    VECMEM_HIP_ERROR_CHECK(hipDeviceSynchronize());
    return true;
}

__global__ void hipJaggedFillKernel(
    vecmem::testing::jagged_soa_container::view view) {

    // Get the thread index.
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Run the modification using the helper function.
    vecmem::testing::jagged_soa_container::device device{view};
    vecmem::testing::fill(i, device);
}

bool hipJaggedFill(vecmem::testing::jagged_soa_container::view view) {

    // Launch the kernel.
    const unsigned int blockSize = 256;
    const unsigned int gridSize = (view.capacity() + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(hipJaggedFillKernel, gridSize, blockSize, 0, 0, view);

    // Check whether it succeeded to run.
    VECMEM_HIP_ERROR_CHECK(hipGetLastError());
    VECMEM_HIP_ERROR_CHECK(hipDeviceSynchronize());
    return true;
}

__global__ void hipSimpleModifyKernel(
    vecmem::testing::simple_soa_container::view view) {

    // Get the thread index.
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Run the modification using the helper function.
    vecmem::testing::simple_soa_container::device device{view};
    vecmem::testing::modify(i, device);
}

bool hipSimpleModify(vecmem::testing::simple_soa_container::view view) {

    // Launch the kernel.
    const unsigned int blockSize = 256;
    const unsigned int gridSize = (view.capacity() + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(hipSimpleModifyKernel, gridSize, blockSize, 0, 0, view);

    // Check whether it succeeded to run.
    VECMEM_HIP_ERROR_CHECK(hipGetLastError());
    VECMEM_HIP_ERROR_CHECK(hipDeviceSynchronize());
    return true;
}

__global__ void hipJaggedModifyKernel(
    vecmem::testing::jagged_soa_container::view view) {

    // Get the thread index.
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Run the modification using the helper function.
    vecmem::testing::jagged_soa_container::device device{view};
    vecmem::testing::modify(i, device);
}

bool hipJaggedModify(vecmem::testing::jagged_soa_container::view view) {

    // Launch the kernel.
    const unsigned int blockSize = 256;
    const unsigned int gridSize = (view.capacity() + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(hipJaggedModifyKernel, gridSize, blockSize, 0, 0, view);

    // Check whether it succeeded to run.
    VECMEM_HIP_ERROR_CHECK(hipGetLastError());
    VECMEM_HIP_ERROR_CHECK(hipDeviceSynchronize());
    return true;
}
