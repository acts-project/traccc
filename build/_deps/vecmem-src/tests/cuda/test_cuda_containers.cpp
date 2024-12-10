/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "test_cuda_containers_kernels.cuh"
#include "vecmem/containers/array.hpp"
#include "vecmem/containers/data/jagged_vector_buffer.hpp"
#include "vecmem/containers/data/vector_buffer.hpp"
#include "vecmem/containers/static_array.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/cuda/device_memory_resource.hpp"
#include "vecmem/memory/cuda/host_memory_resource.hpp"
#include "vecmem/memory/cuda/managed_memory_resource.hpp"
#include "vecmem/utils/cuda/async_copy.hpp"
#include "vecmem/utils/cuda/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

/// Test fixture for the on-device vecmem container tests
class cuda_containers_test : public testing::Test {

protected:
    /// Helper object for performing memory copies
    vecmem::cuda::copy m_copy;

};  // class cuda_containers_test

/// Test a linear transformation using the managed memory resource
TEST_F(cuda_containers_test, managed_memory) {

    // The managed memory resource.
    vecmem::cuda::managed_memory_resource managed_resource;

    // Create an input and an output vector in managed memory.
    vecmem::vector<int> inputvec({1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                 &managed_resource);
    vecmem::vector<int> outputvec(inputvec.size(), &managed_resource);
    EXPECT_EQ(inputvec.size(), outputvec.size());

    // Create the array that is used in the linear transformation.
    vecmem::array<int, 2> constants(managed_resource);
    constants[0] = 2;
    constants[1] = 3;

    // Perform a linear transformation using the vecmem vector helper types.
    linearTransform(vecmem::get_data(constants), vecmem::get_data(inputvec),
                    vecmem::get_data(outputvec));

    // Check the output.
    EXPECT_EQ(inputvec.size(), outputvec.size());
    for (std::size_t i = 0; i < outputvec.size(); ++i) {
        EXPECT_EQ(outputvec.at(i),
                  inputvec.at(i) * constants[0] + constants[1]);
    }
}

/// Test a linear transformation while hand-managing the memory copies
TEST_F(cuda_containers_test, explicit_memory) {

    // The host/device memory resources.
    vecmem::cuda::device_memory_resource device_resource;
    vecmem::cuda::host_memory_resource host_resource;

    // Create input/output vectors on the host.
    vecmem::vector<int> inputvec({1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                 &host_resource);
    vecmem::vector<int> outputvec(inputvec.size(), &host_resource);
    EXPECT_EQ(inputvec.size(), outputvec.size());

    // Allocate a device memory block for the output container.
    auto outputvechost = vecmem::get_data(outputvec);
    vecmem::data::vector_buffer<int> outputvecdevice(
        static_cast<vecmem::data::vector_buffer<int>::size_type>(
            outputvec.size()),
        device_resource);

    // Create the array that is used in the linear transformation.
    vecmem::array<int, 2> constants(host_resource);
    constants[0] = 2;
    constants[1] = 3;

    // Perform a linear transformation with explicit memory copies.
    linearTransform(m_copy.to(vecmem::get_data(constants), device_resource,
                              vecmem::copy::type::host_to_device),
                    m_copy.to(vecmem::get_data(inputvec), device_resource),
                    outputvecdevice);
    m_copy(outputvecdevice, outputvechost, vecmem::copy::type::device_to_host)
        ->wait();

    // Check the output.
    EXPECT_EQ(inputvec.size(), outputvec.size());
    for (std::size_t i = 0; i < outputvec.size(); ++i) {
        EXPECT_EQ(outputvec.at(i),
                  inputvec.at(i) * constants[0] + constants[1]);
    }
}

/// Test a linear transformation while hand-managing the asynchronous memory
/// copies
TEST_F(cuda_containers_test, async_memory) {

    // The host/device memory resources.
    vecmem::cuda::device_memory_resource device_resource;
    vecmem::cuda::host_memory_resource host_resource;

    // The copy utility.
    vecmem::cuda::stream_wrapper stream;
    vecmem::cuda::async_copy copy(stream);

    // Create input/output vectors on the host.
    vecmem::vector<int> inputvec({1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                 &host_resource);
    vecmem::vector<int> outputvec(inputvec.size(), &host_resource);
    EXPECT_EQ(inputvec.size(), outputvec.size());

    // Allocate a device memory block for the output container.
    auto outputvechost = vecmem::get_data(outputvec);
    vecmem::data::vector_buffer<int> outputvecdevice(
        static_cast<vecmem::data::vector_buffer<int>::size_type>(
            outputvec.size()),
        device_resource);

    // Create the array that is used in the linear transformation.
    vecmem::array<int, 2> constants(host_resource);
    constants[0] = 2;
    constants[1] = 3;

    // Perform a linear transformation with explicit memory copies.
    linearTransform(copy.to(vecmem::get_data(constants), device_resource,
                            vecmem::copy::type::host_to_device),
                    copy.to(vecmem::get_data(inputvec), device_resource),
                    outputvecdevice, stream);
    copy(outputvecdevice, outputvechost, vecmem::copy::type::device_to_host)
        ->ignore();
    stream.synchronize();

    // Check the output.
    EXPECT_EQ(inputvec.size(), outputvec.size());
    for (std::size_t i = 0; i < outputvec.size(); ++i) {
        EXPECT_EQ(outputvec.at(i),
                  inputvec.at(i) * constants[0] + constants[1]);
    }
}

/// Test the execution of atomic operations in managed memory
TEST_F(cuda_containers_test, atomic_managed_memory) {

    // The memory resource(s).
    vecmem::cuda::managed_memory_resource resource;

    // Create a small vector in managed memory.
    vecmem::vector<int> vec(100, 0, &resource);

    // Give it to the test function.
    static constexpr unsigned int ITERATIONS = 100;
    atomicTransform(ITERATIONS, vecmem::get_data(vec));

    // Check the output.
    for (int value : vec) {
        EXPECT_EQ(static_cast<unsigned int>(value), 4 * ITERATIONS);
    }
}

/// Test the execution of atomic operations in device memory
TEST_F(cuda_containers_test, atomic_device_memory) {

    // The memory resources.
    vecmem::cuda::host_memory_resource host_resource;
    vecmem::cuda::device_memory_resource device_resource;

    // Create a small vector in host memory.
    vecmem::vector<int> vec(100, 0, &host_resource);

    // Copy it to the device.
    auto vec_on_device = m_copy.to(vecmem::get_data(vec), device_resource);

    // Give it to the test function.
    static constexpr unsigned int ITERATIONS = 100;
    atomicTransform(ITERATIONS, vec_on_device);

    // Copy it back to the host.
    m_copy(vec_on_device, vec)->wait();

    // Check the output.
    for (int value : vec) {
        EXPECT_EQ(static_cast<unsigned int>(value), 4 * ITERATIONS);
    }
}

/// Test the execution of atomic operations in local memory
TEST_F(cuda_containers_test, atomic_local_ref) {

    // The memory resources.
    vecmem::cuda::host_memory_resource host_resource;
    vecmem::cuda::device_memory_resource device_resource;

    // Local block size.
    static constexpr int BLOCKSIZE = 128;

    // Number of blocks.
    static constexpr int NUMBLOCKS = 5;

    // Allocate memory on the host, and set initial values in it.
    vecmem::vector<int> host_vector(NUMBLOCKS, 0, &host_resource);

    // Set up device buffers with the data.
    auto device_buffer =
        m_copy.to(vecmem::get_data(host_vector), device_resource);

    // Run test function.
    atomicLocalRef(NUMBLOCKS, BLOCKSIZE, device_buffer);

    // Copy data back to the host.
    m_copy(device_buffer, host_vector)->wait();

    // Check the output.
    for (std::size_t i = 0; i < NUMBLOCKS; ++i) {
        EXPECT_EQ(host_vector[i], i * BLOCKSIZE);
    }
}

/// Test the usage of extendable vectors in a kernel
TEST_F(cuda_containers_test, extendable_memory) {

    // The memory resources.
    vecmem::cuda::managed_memory_resource managed_resource;
    vecmem::cuda::device_memory_resource device_resource;
    vecmem::cuda::host_memory_resource host_resource;

    // Create a small (input) vector in managed memory.
    vecmem::vector<int> input(&managed_resource);
    for (int i = 0; i < 100; ++i) {
        input.push_back(i);
    }

    // Create a buffer that will hold the filtered elements of the input vector.
    vecmem::data::vector_buffer<int> output_buffer(
        static_cast<vecmem::data::vector_buffer<int>::size_type>(input.size()),
        device_resource, vecmem::data::buffer_type::resizable);
    m_copy.setup(output_buffer)->wait();

    // Run the filtering kernel.
    filterTransform(vecmem::get_data(input), output_buffer);

    // Copy the output into the host's memory.
    vecmem::vector<int> output(&host_resource);
    m_copy(output_buffer, output)->wait();

    // Check its contents.
    EXPECT_EQ(output.size(), 89);
    for (int value : output) {
        EXPECT_LT(10, value);
    }
}

/// Test the usage of an @c array<vector<...>> construct
TEST_F(cuda_containers_test, array_memory) {

    // The memory resource(s).
    vecmem::cuda::managed_memory_resource managed_resource;

    // Create an array of vectors.
    vecmem::static_array<vecmem::vector<int>, 4> vec_array{
        vecmem::vector<int>{{1, 2, 3, 4}, &managed_resource},
        vecmem::vector<int>{{5, 6}, &managed_resource},
        vecmem::vector<int>{{7, 8, 9}, &managed_resource},
        vecmem::vector<int>{&managed_resource}};

    // Create an appropriate data object out of it.
    vecmem::static_array<vecmem::data::vector_view<int>, 4> vec_data{
        vecmem::get_data(vec_array[0]), vecmem::get_data(vec_array[1]),
        vecmem::get_data(vec_array[2]), vecmem::get_data(vec_array[3])};

    // Run a kernel on it.
    arrayTransform(vec_data);

    // Check its contents.
    EXPECT_EQ(vec_array.at(0).at(0), 2);
    EXPECT_EQ(vec_array.at(0).at(1), 4);
    EXPECT_EQ(vec_array.at(0).at(2), 6);
    EXPECT_EQ(vec_array.at(0).at(3), 8);
    EXPECT_EQ(vec_array.at(1).at(0), 10);
    EXPECT_EQ(vec_array.at(1).at(1), 12);
    EXPECT_EQ(vec_array.at(2).at(0), 14);
    EXPECT_EQ(vec_array.at(2).at(1), 16);
    EXPECT_EQ(vec_array.at(2).at(2), 18);
    EXPECT_EQ(vec_array.at(3).size(), 0u);
}

/// Test buffers with "large" elements (for which alignment becomes important)
TEST_F(cuda_containers_test, large_buffer) {

    // The memory resource(s).
    vecmem::cuda::managed_memory_resource managed_resource;

    // Test a (1D) vector.
    vecmem::data::vector_buffer<unsigned long> buffer1(
        100, managed_resource, vecmem::data::buffer_type::resizable);
    m_copy.setup(buffer1)->wait();
    largeBufferTransform(buffer1);
    EXPECT_EQ(m_copy.get_size(buffer1), 21u);

    // Test a (2D) jagged vector.
    vecmem::data::jagged_vector_buffer<unsigned long> buffer2(
        {100, 100, 100}, managed_resource, nullptr,
        vecmem::data::buffer_type::resizable);
    m_copy.setup(buffer2)->wait();
    largeBufferTransform(buffer2);
    EXPECT_EQ(m_copy.get_sizes(buffer2),
              std::vector<unsigned int>({5u, 21u, 10u}));
}
