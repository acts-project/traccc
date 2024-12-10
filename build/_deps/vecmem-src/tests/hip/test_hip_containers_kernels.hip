/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../hip/src/utils/hip_error_handling.hpp"
#include "test_hip_containers_kernels.hpp"
#include "vecmem/containers/device_array.hpp"
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/containers/jagged_device_vector.hpp"
#include "vecmem/containers/static_array.hpp"
#include "vecmem/memory/atomic.hpp"
#include "vecmem/memory/device_atomic_ref.hpp"
#include "vecmem/utils/tuple.hpp"

// HIP include(s).
#include <hip/hip_runtime.h>

/// Kernel performing a linear transformation using the vector helper types
__global__ void linearTransformKernel(
    vecmem::data::vector_view<const int> constants,
    vecmem::data::vector_view<const int> input,
    vecmem::data::vector_view<int> output) {

    // Find the current index.
    const std::size_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
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
    hipLaunchKernelGGL(linearTransformKernel, 1, input.size(), 0, nullptr,
                       constants, input, output);
    // Check whether it succeeded to run.
    VECMEM_HIP_ERROR_CHECK(hipGetLastError());
    VECMEM_HIP_ERROR_CHECK(hipDeviceSynchronize());
}

/// Kernel performing a linear transformation using the vector helper types
__global__ void linearTransformKernel(
    vecmem::data::vector_view<const int> constants,
    vecmem::data::jagged_vector_view<const int> input,
    vecmem::data::jagged_vector_view<int> output) {

    // Find the current index.
    const std::size_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
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

void linearTransform(vecmem::data::vector_view<const int> constants,
                     vecmem::data::jagged_vector_view<const int> input,
                     vecmem::data::jagged_vector_view<int> output) {

    // A sanity check.
    assert(input.size() == output.size());

    // Launch the kernel.
    hipLaunchKernelGGL(linearTransformKernel, 1, input.size(), 0, nullptr,
                       constants, input, output);
    // Check whether it succeeded to run.
    VECMEM_HIP_ERROR_CHECK(hipGetLastError());
    VECMEM_HIP_ERROR_CHECK(hipDeviceSynchronize());
}

/// Kernel performing some basic atomic operations.
__global__ void atomicTransformKernel(std::size_t iterations,
                                      vecmem::data::vector_view<int> data) {

    // Find the current global index.
    const std::size_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
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

void atomicTransform(std::size_t iterations,
                     vecmem::data::vector_view<int> vec) {

    // Launch the kernel.
    hipLaunchKernelGGL(atomicTransformKernel, iterations, vec.size(), 0,
                       nullptr, iterations, vec);
    // Check whether it succeeded to run.
    VECMEM_HIP_ERROR_CHECK(hipGetLastError());
    VECMEM_HIP_ERROR_CHECK(hipDeviceSynchronize());
}

/// Kernel performing some basic atomic operations on local memory.
__global__ void atomicLocalRefKernel(vecmem::data::vector_view<int> data) {

    // Find the current block index.
    const int i = hipBlockIdx_x;

    __shared__ int shared;

    // Initialise shared memory variable
    if (hipThreadIdx_x == 0) {
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
    if (hipThreadIdx_x == 0) {
        vecmem::device_vector<int> dev(data);
        dev.at(i) = shared;
    }

    return;
}

void atomicLocalRef(std::size_t num_blocks, std::size_t block_size,
                    vecmem::data::vector_view<int> vec) {

    // Launch the kernel.
    hipLaunchKernelGGL(atomicLocalRefKernel, num_blocks, block_size, 0, nullptr,
                       vec);
    // Check whether it succeeded to run.
    VECMEM_HIP_ERROR_CHECK(hipGetLastError());
    VECMEM_HIP_ERROR_CHECK(hipDeviceSynchronize());
}

/// Kernel filtering the input vector elements into the output vector
__global__ void filterTransformKernel(
    vecmem::data::vector_view<const int> input,
    vecmem::data::vector_view<int> output) {

    // Find the current global index.
    const std::size_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
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
    hipLaunchKernelGGL(filterTransformKernel, 1, input.size(), 0, nullptr,
                       input, output);
    // Check whether it succeeded to run.
    VECMEM_HIP_ERROR_CHECK(hipGetLastError());
    VECMEM_HIP_ERROR_CHECK(hipDeviceSynchronize());
}

/// Kernel filtering the input vector elements into the output vector
__global__ void filterTransformKernel(
    vecmem::data::jagged_vector_view<const int> input,
    vecmem::data::jagged_vector_view<int> output) {

    // Find the current indices.
    const std::size_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (i >= input.size()) {
        return;
    }
    const std::size_t j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
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
                     std::size_t max_vec_size,
                     vecmem::data::jagged_vector_view<int> output) {

    // Launch the kernel.
    hipLaunchKernelGGL(filterTransformKernel, 1,
                       dim3(input.size(), max_vec_size), 0, nullptr, input,
                       output);
    // Check whether it succeeded to run.
    VECMEM_HIP_ERROR_CHECK(hipGetLastError());
    VECMEM_HIP_ERROR_CHECK(hipDeviceSynchronize());
}

/// Kernel filling a jagged vector to its capacity
__global__ void fillTransformKernel(
    vecmem::data::jagged_vector_view<int> vec_data) {

    // Find the current index.
    const std::size_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
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
    hipLaunchKernelGGL(fillTransformKernel, vec.size(), 1, 0, nullptr, vec);

    // Check whether it succeeded to run.
    VECMEM_HIP_ERROR_CHECK(hipGetLastError());
    VECMEM_HIP_ERROR_CHECK(hipDeviceSynchronize());
}

/// Kernel multiplying each element of the received structure by 2.
__global__ void arrayTransformKernel(
    vecmem::static_array<vecmem::data::vector_view<int>, 4> data) {

    // Find the current indices,
    const std::size_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const std::size_t j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
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
    hipLaunchKernelGGL(arrayTransformKernel, 1, dimensions, 0, nullptr, data);

    // Check whether it succeeded to run.
    VECMEM_HIP_ERROR_CHECK(hipGetLastError());
    VECMEM_HIP_ERROR_CHECK(hipDeviceSynchronize());
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
    hipLaunchKernelGGL(largeBufferTransformKernel, 1, 1, 0, nullptr, data);

    // Check whether it succeeded to run.
    VECMEM_HIP_ERROR_CHECK(hipGetLastError());
    VECMEM_HIP_ERROR_CHECK(hipDeviceSynchronize());
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
    hipLaunchKernelGGL(largeBufferTransformKernel, 1, 1, 0, nullptr, data);

    // Check whether it succeeded to run.
    VECMEM_HIP_ERROR_CHECK(hipGetLastError());
    VECMEM_HIP_ERROR_CHECK(hipDeviceSynchronize());
}
