/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../common/jagged_soa_container_helpers.hpp"
#include "../common/simple_soa_container_helpers.hpp"
#include "test_cuda_edm_kernels.hpp"

// Project include(s).
#include "../../cuda/src/utils/cuda_error_handling.hpp"

__global__ void cudaSimpleFillKernel(
    vecmem::testing::simple_soa_container::view view) {

    // Get the thread index.
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Run the modification using the helper function.
    vecmem::testing::simple_soa_container::device device{view};
    vecmem::testing::fill(i, device);
}

bool cudaSimpleFill(vecmem::testing::simple_soa_container::view view) {

    // Launch the kernel.
    const unsigned int blockSize = 256;
    const unsigned int gridSize = (view.capacity() + blockSize - 1) / blockSize;
    cudaSimpleFillKernel<<<gridSize, blockSize>>>(view);

    // Check whether it succeeded to run.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    return true;
}

__global__ void cudaJaggedFillKernel(
    vecmem::testing::jagged_soa_container::view view) {

    // Get the thread index.
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Run the modification using the helper function.
    vecmem::testing::jagged_soa_container::device device{view};
    vecmem::testing::fill(i, device);
}

bool cudaJaggedFill(vecmem::testing::jagged_soa_container::view view) {

    // Launch the kernel.
    const unsigned int blockSize = 256;
    const unsigned int gridSize = (view.capacity() + blockSize - 1) / blockSize;
    cudaJaggedFillKernel<<<gridSize, blockSize>>>(view);

    // Check whether it succeeded to run.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    return true;
}

__global__ void cudaSimpleModifyKernel(
    vecmem::testing::simple_soa_container::view view) {

    // Get the thread index.
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Run the modification using the helper function.
    vecmem::testing::simple_soa_container::device device{view};
    vecmem::testing::modify(i, device);
}

bool cudaSimpleModify(vecmem::testing::simple_soa_container::view view) {

    // Launch the kernel.
    const unsigned int blockSize = 256;
    const unsigned int gridSize = (view.capacity() + blockSize - 1) / blockSize;
    cudaSimpleModifyKernel<<<gridSize, blockSize>>>(view);

    // Check whether it succeeded to run.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    return true;
}

__global__ void cudaJaggedModifyKernel(
    vecmem::testing::jagged_soa_container::view view) {

    // Get the thread index.
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Run the modification using the helper function.
    vecmem::testing::jagged_soa_container::device device{view};
    vecmem::testing::modify(i, device);
}

bool cudaJaggedModify(vecmem::testing::jagged_soa_container::view view) {

    // Launch the kernel.
    const unsigned int blockSize = 256;
    const unsigned int gridSize = (view.capacity() + blockSize - 1) / blockSize;
    cudaJaggedModifyKernel<<<gridSize, blockSize>>>(view);

    // Check whether it succeeded to run.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    return true;
}
