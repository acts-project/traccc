/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../cuda/src/utils/cuda_error_handling.hpp"
#include "test_cuda_jagged_vector_view_kernels.cuh"
#include "vecmem/containers/device_array.hpp"
#include "vecmem/containers/jagged_device_vector.hpp"

// System include(s).
#include <cassert>

/// Kernel performing a linear transformation using the vector helper types
__global__ void linearTransformKernel(
    vecmem::data::vector_view<const int> constants,
    vecmem::data::jagged_vector_view<const int> input,
    vecmem::data::jagged_vector_view<int> output) {

    // Find the current index.
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= input.size()) {
        return;
    }

    // Create the helper containers.
    const vecmem::device_array<const int, 2> constantarray(constants);
    const vecmem::jagged_device_vector<const int> inputvec(input);
    vecmem::jagged_device_vector<int> outputvec(output);

    // A little sanity check.
    assert(inputvec.at(i).size() == outputvec.at(i).size());

    // Perform the requested linear transformation on all elements of a given
    // "internal vector".
    for (std::size_t j = 0; j < inputvec[i].size(); ++j) {
        outputvec[i][j] = inputvec[i][j] * constantarray[0] + constantarray[1];
    }
    __syncthreads();

    // Now exercise the jagged vector iterators in a bit of an elaborate
    // operation.
    for (auto itr1 = outputvec.rbegin(); itr1 != outputvec.rend(); ++itr1) {
        if ((outputvec[i].size() > 0) && (itr1->size() > 1)) {
            // Iterate over all inner vectors, skipping the first elements.
            // Since those are being updated at the same time, by other threads.
            for (auto itr2 = itr1->rbegin(); itr2 != (itr1->rend() - 1);
                 ++itr2) {
                outputvec[i].front() += *itr2;
            }
        }
    }
}

void linearTransform(const vecmem::data::vector_view<int>& constants,
                     const vecmem::data::jagged_vector_view<int>& input,
                     vecmem::data::jagged_vector_view<int>& output) {

    // A sanity check.
    assert(input.size() == output.size());

    // Launch the kernel.
    linearTransformKernel<<<1, static_cast<unsigned int>(input.size())>>>(
        constants, input, output);
    // Check whether it succeeded to run.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}
