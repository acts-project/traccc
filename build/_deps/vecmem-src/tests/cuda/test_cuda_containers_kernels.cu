/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../cuda/src/utils/cuda_error_handling.hpp"
#include "../../cuda/src/utils/cuda_wrappers.hpp"
#include "test_cuda_containers_kernels.cuh"
#include "vecmem/containers/device_array.hpp"
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/containers/jagged_device_vector.hpp"
#include "vecmem/containers/static_array.hpp"
#include "vecmem/memory/atomic.hpp"
#include "vecmem/memory/device_atomic_ref.hpp"
#include "vecmem/utils/tuple.hpp"

// System include(s).
#include <cassert>

/// Kernel performing a linear transformation using the vector helper types
__global__ void linearTransformKernel(
    vecmem::data::vector_view<const int> constants,
    vecmem::data::vector_view<const int> input,
    vecmem::data::vector_view<int> output) {

    // Find the current index.
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= input.size()) {
        return;
    }

    // Create the helper containers.
    const vecmem::device_array<const int, 2> constantarray1(constants);
    const vecmem::static_array<int, 2> constantarray2 = {constantarray1[0],
                                                         constantarray1[1]};
    auto tuple1 = vecmem::make_tuple(constantarray1[0], constantarray1[1]);
    auto tuple2 = vecmem::tie(constantarray1, constantarray2);
    const vecmem::device_vector<const int> inputvec(input);
    vecmem::device_vector<int> outputvec(output);

    // Perform the linear transformation.
    outputvec.at(i) = inputvec.at(i) * constantarray1.at(0) +
                      vecmem::get<1>(constantarray2) + vecmem::get<0>(tuple1) -
                      vecmem::get<1>(tuple2)[0];
    return;
}

void linearTransform(vecmem::data::vector_view<const int> constants,
                     vecmem::data::vector_view<const int> input,
                     vecmem::data::vector_view<int> output) {

    // Launch the kernel.
    linearTransformKernel<<<1, input.size()>>>(constants, input, output);
    // Check whether it succeeded to run.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

void linearTransform(vecmem::data::vector_view<const int> constants,
                     vecmem::data::vector_view<const int> input,
                     vecmem::data::vector_view<int> output,
                     const vecmem::cuda::stream_wrapper& stream) {

    // Launch the kernel.
    linearTransformKernel<<<1, input.size(), 0,
                            vecmem::cuda::details::get_stream(stream)>>>(
        constants, input, output);
    // Check whether it succeeded to launch.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
}

/// Kernel performing some basic atomic operations.
__global__ void atomicTransformKernel(std::size_t iterations,
                                      vecmem::data::vector_view<int> data) {

    // Find the current global index.
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (data.size() * iterations)) {
        return;
    }

    // Get a pointer to the integer that this thread will work on.
    const std::size_t array_index = i % data.size();
    assert(array_index < data.size());
    int* ptr = data.ptr() + array_index;

    // Do some simple stuff with it using vecmem::atomic.
    vecmem::atomic<int> a(ptr);
    a.fetch_add(4);
    a.fetch_sub(2);
    a.fetch_and(0xffffffff);
    a.fetch_or(0x00000000);

    // Do the same simple stuff with it using vecmem::atomic_ref.
    vecmem::device_atomic_ref<int> a2(*ptr);
    a2.fetch_add(4);
    a2.fetch_sub(2);
    a2.fetch_and(0xffffffff);
    a2.fetch_or(0x00000000);
    return;
}

void atomicTransform(unsigned int iterations,
                     vecmem::data::vector_view<int> vec) {

    // Launch the kernel.
    atomicTransformKernel<<<iterations, vec.size()>>>(iterations, vec);
    // Check whether it succeeded to run.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

/// Kernel performing some basic atomic operations on local memory.
__global__ void atomicLocalRefKernel(vecmem::data::vector_view<int> data) {

    // Find the current block index.
    const int i = blockIdx.x;

    __shared__ int shared;

    // Initialise shared memory variable
    if (threadIdx.x == 0) {
        shared = 0;
    }
    __syncthreads();

    // Perform basic atomic operations on local memory.
    vecmem::device_atomic_ref<int, vecmem::device_address_space::local> atom(
        shared);
    atom.fetch_add(2 * i);
    atom.fetch_sub(i);
    atom.fetch_and(0xffffffff);
    atom.fetch_or(0x00000000);
    __syncthreads();
    if (threadIdx.x == 0) {
        vecmem::device_vector<int> dev(data);
        dev.at(i) = shared;
    }

    return;
}

void atomicLocalRef(unsigned int num_blocks, unsigned int block_size,
                    vecmem::data::vector_view<int> vec) {

    // Launch the kernel.
    atomicLocalRefKernel<<<num_blocks, block_size>>>(vec);
    // Check whether it succeeded to run.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

/// Kernel filtering the input vector elements into the output vector
__global__ void filterTransformKernel(
    vecmem::data::vector_view<const int> input,
    vecmem::data::vector_view<int> output) {

    // Find the current global index.
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= input.size()) {
        return;
    }

    // Set up the vector objects.
    const vecmem::device_vector<const int> inputvec(input);
    vecmem::device_vector<int> outputvec(output);

    // Add this thread's element, if it passes the selection.
    const int element = inputvec.at(i);
    if (element > 10) {
        outputvec.push_back(element);
    }
    return;
}

void filterTransform(vecmem::data::vector_view<const int> input,
                     vecmem::data::vector_view<int> output) {

    // Launch the kernel.
    filterTransformKernel<<<1, input.size()>>>(input, output);
    // Check whether it succeeded to run.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

/// Kernel filtering the input vector elements into the output vector
__global__ void filterTransformKernel(
    vecmem::data::jagged_vector_view<const int> input,
    vecmem::data::jagged_vector_view<int> output) {

    // Find the current indices.
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= input.size()) {
        return;
    }
    const std::size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= input.ptr()[i].size()) {
        return;
    }

    // Set up the vector objects.
    const vecmem::jagged_device_vector<const int> inputvec(input);
    vecmem::jagged_device_vector<int> outputvec(output);

    // Keep just the odd elements.
    const int value = inputvec[i][j];
    if ((value % 2) != 0) {
        outputvec.at(i).push_back(value);
    }
    return;
}

void filterTransform(vecmem::data::jagged_vector_view<const int> input,
                     unsigned int max_vec_size,
                     vecmem::data::jagged_vector_view<int> output) {

    // Launch the kernel.
    dim3 dimensions(static_cast<unsigned int>(input.size()), max_vec_size);
    filterTransformKernel<<<1, dimensions>>>(input, output);
    // Check whether it succeeded to run.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

/// Kernel filling a jagged vector to its capacity
__global__ void fillTransformKernel(
    vecmem::data::jagged_vector_view<int> vec_data) {

    // Find the current index.
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vec_data.size()) {
        return;
    }

    // Create a device vector on top of the view.
    vecmem::jagged_device_vector<int> vec(vec_data);

    // Fill the vectors to their capacity.
    while (vec[i].size() < vec[i].capacity()) {
        vec[i].push_back(1);
    }
}

void fillTransform(vecmem::data::jagged_vector_view<int> vec) {

    // Launch the kernel
    fillTransformKernel<<<static_cast<unsigned int>(vec.size()), 1>>>(vec);

    // Check whether it succeeded to run.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

/// Kernel multiplying each element of the received structure by 2.
__global__ void arrayTransformKernel(
    vecmem::static_array<vecmem::data::vector_view<int>, 4> data) {

    // Find the current indices,
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= data.size()) {
        return;
    }
    if (j >= data[i].size()) {
        return;
    }

    // Create the "device type".
    vecmem::static_array<vecmem::device_vector<int>, 4> vec{
        vecmem::device_vector<int>{data[0]},
        vecmem::device_vector<int>{data[1]},
        vecmem::device_vector<int>{data[2]},
        vecmem::device_vector<int>{data[3]}};

    // Perform the transformation.
    vec[i][j] *= 2;
}

void arrayTransform(
    vecmem::static_array<vecmem::data::vector_view<int>, 4> data) {

    // Launch the kernel.
    const dim3 dimensions(4u, 4u);
    arrayTransformKernel<<<1, dimensions>>>(data);

    // Check whether it succeeded to run.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

/// Kernel making a trivial use of the resizable vector that it receives
__global__ void largeBufferTransformKernel(
    vecmem::data::vector_view<unsigned long> data) {

    // Add one element to the vector in just the first thread
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i != 0) {
        return;
    }
    vecmem::device_vector<unsigned long> vec(data);
    assert(vec.size() == 0);
    vec.push_back(0);
    vec.bulk_append(5);
    vec.bulk_append(5, 2);
    vec.bulk_append_implicit(5);
    vec.bulk_append_implicit_unsafe(5);
}

void largeBufferTransform(vecmem::data::vector_view<unsigned long> data) {

    // Launch the kernel.
    largeBufferTransformKernel<<<1, 1>>>(data);

    // Check whether it succeeded to run.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

/// Kernel making a trivial use of the resizable jagged vector that it receives
__global__ void largeBufferTransformKernel(
    vecmem::data::jagged_vector_view<unsigned long> data) {

    // Add one element to the vector in just the first thread
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i != 0) {
        return;
    }
    vecmem::jagged_device_vector<unsigned long> vec(data);
    assert(vec.size() == 3);
    assert(vec.at(1).size() == 0);
    vec.at(0).resize_implicit(5);
    vec.at(1).push_back(0);
    vec.at(1).bulk_append(5);
    vec.at(1).bulk_append(5, 2);
    vec.at(1).bulk_append_implicit(5);
    vec.at(1).bulk_append_implicit_unsafe(5);
    vec.at(2).resize_implicit_unsafe(10);
}

void largeBufferTransform(
    vecmem::data::jagged_vector_view<unsigned long> data) {

    // Launch the kernel.
    largeBufferTransformKernel<<<1, 1>>>(data);

    // Check whether it succeeded to run.
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}
