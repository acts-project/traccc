/**
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include <../../device/cuda/src/utils/sort.cuh>

__global__ void initializeArrayKernel(uint32_t *keys, uint32_t n_keys) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t i = tid; i < n_keys; i += blockDim.x) {
        keys[i] = (13 * i) % n_keys;
    }
}

__global__ void testBlockSortKernel(uint32_t *keys, uint32_t n_keys) {
    traccc::cuda::blockOddEvenKeySort(keys, n_keys, std::less<uint32_t>());
}

__global__ void testWarpSortKernel(uint32_t *keys, uint32_t n_keys) {
    traccc::cuda::warpOddEvenKeySort(keys, n_keys, std::less<uint32_t>());
}

TEST(CUDASort, BlockOddEvenKeySort) {
    uint32_t n = 2803;
    uint32_t *dev_arr = nullptr;
    std::unique_ptr<uint32_t[]> host_arr = std::make_unique<uint32_t[]>(n);

    ASSERT_EQ(cudaMalloc(&dev_arr, n * sizeof(uint32_t)), cudaSuccess);
    ASSERT_NE(dev_arr, nullptr);

    initializeArrayKernel<<<1, 1024u>>>(dev_arr, n);
    ASSERT_EQ(cudaPeekAtLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    testBlockSortKernel<<<1, 1024u>>>(dev_arr, n);

    ASSERT_EQ(cudaPeekAtLastError(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(host_arr.get(), dev_arr, n * sizeof(uint32_t),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    for (uint32_t i = 0; i < n; ++i) {
        ASSERT_EQ(host_arr[i], i);
    }

    ASSERT_EQ(cudaFree(dev_arr), cudaSuccess);
}

TEST(CUDASort, WarpOddEvenKeySort) {
    uint32_t n = 2803;
    uint32_t *dev_arr = nullptr;
    std::unique_ptr<uint32_t[]> host_arr = std::make_unique<uint32_t[]>(n);

    ASSERT_EQ(cudaMalloc(&dev_arr, n * sizeof(uint32_t)), cudaSuccess);
    ASSERT_NE(dev_arr, nullptr);

    initializeArrayKernel<<<1, 1024u>>>(dev_arr, n);
    ASSERT_EQ(cudaPeekAtLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    testWarpSortKernel<<<1, 32u>>>(dev_arr, n);

    ASSERT_EQ(cudaPeekAtLastError(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(host_arr.get(), dev_arr, n * sizeof(uint32_t),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    for (uint32_t i = 0; i < n; ++i) {
        ASSERT_EQ(host_arr[i], i);
    }

    ASSERT_EQ(cudaFree(dev_arr), cudaSuccess);
}
