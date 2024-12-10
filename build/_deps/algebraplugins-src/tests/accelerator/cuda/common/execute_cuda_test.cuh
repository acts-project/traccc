/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Test include(s).
#include "cuda_error_check.hpp"

// System include(s).
#include <cstddef>

namespace {

template <class FUNCTOR, typename... ARGS>
__global__ void cudaTestKernel(std::size_t arraySizes, ARGS... args) {

  // Find the current index that we need to process.
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= arraySizes) {
    return;
  }

  // Execute the test functor for this index.
  FUNCTOR()(i, args...);
  return;
}

}  // namespace

/// Execute a test functor on a device, on @c arraySizes threads
template <class FUNCTOR, class... ARGS>
void execute_cuda_test(std::size_t arraySizes, ARGS... args) {

  // Number of threads per execution block. Less than 1024 to make debug tests
  // possible.
  int nThreadsPerBlock = 256;

  // If the arrays are not even this large, then reduce the value to the
  // size of the arrays.
  if (arraySizes < nThreadsPerBlock) {
    nThreadsPerBlock = static_cast<int>(arraySizes);
  }

  // Launch the test on the device.
  const int nBlocks =
      static_cast<int>((arraySizes + nThreadsPerBlock - 1) / nThreadsPerBlock);
  cudaTestKernel<FUNCTOR><<<nBlocks, nThreadsPerBlock>>>(arraySizes, args...);

  // Check whether it succeeded to run.
  CUDA_ERROR_CHECK(cudaGetLastError());
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}
