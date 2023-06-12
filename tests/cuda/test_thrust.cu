/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// Thrust include(s).
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <vector>

namespace {
vecmem::cuda::copy copy;
vecmem::host_memory_resource host_resource;
vecmem::cuda::device_memory_resource device_resource;

}  // namespace

TEST(thrust, sort) {

    vecmem::vector<unsigned int> host_vector{{3, 2, 1, 8, 4}, &host_resource};

    auto host_buffer = vecmem::get_data(host_vector);
    auto device_buffer = copy.to(vecmem::get_data(host_vector), device_resource,
                                 vecmem::copy::type::host_to_device);

    vecmem::device_vector<unsigned int> device_vector(device_buffer);

    thrust::sort(thrust::device, device_vector.begin(), device_vector.end());

    copy(device_buffer, host_buffer, vecmem::copy::type::device_to_host);

    ASSERT_EQ(host_vector[0], 1);
    ASSERT_EQ(host_vector[1], 2);
    ASSERT_EQ(host_vector[2], 3);
    ASSERT_EQ(host_vector[3], 4);
    ASSERT_EQ(host_vector[4], 8);
}

TEST(thrust, scan) {

    vecmem::vector<unsigned int> host_vector{{3, 2, 1, 8, 4}, &host_resource};

    auto host_buffer = vecmem::get_data(host_vector);
    auto device_buffer = copy.to(vecmem::get_data(host_vector), device_resource,
                                 vecmem::copy::type::host_to_device);

    vecmem::device_vector<unsigned int> device_vector(device_buffer);

    thrust::inclusive_scan(thrust::device, device_vector.begin(),
                           device_vector.end(), device_vector.begin());

    copy(device_buffer, host_buffer, vecmem::copy::type::device_to_host);

    ASSERT_EQ(host_vector[0], 3);
    ASSERT_EQ(host_vector[1], 5);
    ASSERT_EQ(host_vector[2], 6);
    ASSERT_EQ(host_vector[3], 14);
    ASSERT_EQ(host_vector[4], 18);
}

TEST(thrust, copy) {

    // This unit test is written to validate the commands used in
    // finding_algorith.cu

    // Effective sizes of host input vectors
    std::vector<std::size_t> sizes{1, 3, 2};

    // Host input vectors to be copied
    // {{1},{3,1,4},{1,1}} with effectiv sizes of {1, 3, 2}
    vecmem::vector<unsigned int> host_vec0{{1, 0, 0}, &host_resource};
    vecmem::vector<unsigned int> host_vec1{{3, 1, 4}, &host_resource};
    vecmem::vector<unsigned int> host_vec2{{1, 1, 0}, &host_resource};

    // Put the buffers of vectors into map (H -> D copy)
    std::map<unsigned int, vecmem::data::vector_buffer<unsigned int>> vec_map;

    vec_map[0] = copy.to(vecmem::get_data(host_vec0), device_resource,
                         vecmem::copy::type::host_to_device);

    vec_map[1] = copy.to(vecmem::get_data(host_vec1), device_resource,
                         vecmem::copy::type::host_to_device);

    vec_map[2] = copy.to(vecmem::get_data(host_vec2), device_resource,
                         vecmem::copy::type::host_to_device);

    // Create a buffer to copy the host input vectors
    vecmem::data::jagged_vector_buffer<unsigned int> jagged_buffer(
        sizes, device_resource, &host_resource);

    // Copy the map into the buffer (D->D copy)
    for (unsigned int i = 0; i < 3; i++) {
        vecmem::device_vector<unsigned int> in(vec_map[i]);

        vecmem::device_vector<unsigned int> out(
            *(jagged_buffer.host_ptr() + i));

        thrust::copy(thrust::device, in.begin(), in.begin() + sizes[i],
                     out.begin());
    }

    // Create the final output vector
    vecmem::jagged_vector<unsigned int> jagged_vec(&host_resource);
    jagged_vec.resize(sizes.size());
    jagged_vec[0].resize(sizes[0]);
    jagged_vec[1].resize(sizes[1]);
    jagged_vec[2].resize(sizes[2]);

    // Copy the buffer to vector (D->H copy)
    copy(jagged_buffer, vecmem::get_data(jagged_vec),
         vecmem::copy::type::device_to_host);

    // Result should be {{1},{3,1,4},{1,1}}
    ASSERT_EQ(jagged_vec[0].size(), 1);
    ASSERT_EQ(jagged_vec[0][0], 1);
    ASSERT_EQ(jagged_vec[1].size(), 3);
    ASSERT_EQ(jagged_vec[1][0], 3);
    ASSERT_EQ(jagged_vec[1][1], 1);
    ASSERT_EQ(jagged_vec[1][2], 4);
    ASSERT_EQ(jagged_vec[2].size(), 2);
    ASSERT_EQ(jagged_vec[2][0], 1);
    ASSERT_EQ(jagged_vec[2][1], 1);
}

TEST(thrust, fill) {

    vecmem::vector<unsigned int> host_vector{{1, 1, 1, 1, 1, 1, 1},
                                             &host_resource};

    auto host_buffer = vecmem::get_data(host_vector);
    auto device_buffer = copy.to(vecmem::get_data(host_vector), device_resource,
                                 vecmem::copy::type::host_to_device);

    vecmem::device_vector<unsigned int> device_vector(device_buffer);

    thrust::fill(thrust::device, device_vector.begin(), device_vector.end(),
                 112);

    copy(device_buffer, host_buffer, vecmem::copy::type::device_to_host);

    ASSERT_EQ(host_vector[0], 112);
    ASSERT_EQ(host_vector[1], 112);
    ASSERT_EQ(host_vector[2], 112);
    ASSERT_EQ(host_vector[3], 112);
    ASSERT_EQ(host_vector[4], 112);
    ASSERT_EQ(host_vector[5], 112);
    ASSERT_EQ(host_vector[6], 112);
}