/*
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// vecmem includes
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

// traccc includes
#include <traccc/cuda/sanity/contiguous_on.cuh>
#include <traccc/definitions/qualifiers.hpp>

// GTest include(s).
#include <gtest/gtest.h>

struct int_identity_projection {
    TRACCC_HOST_DEVICE
    int operator()(const int& v) { return v; }
};

class CudaSanityContiguousOn : public testing::Test {
    protected:
    CudaSanityContiguousOn() : copy(stream.cudaStream()) {}

    vecmem::cuda::device_memory_resource mr;
    traccc::cuda::stream stream;
    vecmem::cuda::async_copy copy;
};

TEST_F(CudaSanityContiguousOn, TrueOrdered) {
    std::vector<int> host_vector;

    for (int i = 0; i < 5000; ++i) {
        for (int j = 0; j < i; ++j) {
            host_vector.push_back(i);
        }
    }

    auto device_data = copy.to(vecmem::get_data(host_vector), mr,
                               vecmem::copy::type::host_to_device);

    ASSERT_TRUE(traccc::cuda::is_contiguous_on(int_identity_projection(), mr,
                                               copy, stream, device_data));
}

TEST_F(CudaSanityContiguousOn, TrueRandom) {
    std::vector<int> host_vector;

    for (int i : {603, 6432, 1, 3, 67, 2, 1111}) {
        for (int j = 0; j < i; ++j) {
            host_vector.push_back(i);
        }
    }

    auto device_data = copy.to(vecmem::get_data(host_vector), mr,
                               vecmem::copy::type::host_to_device);

    ASSERT_TRUE(traccc::cuda::is_contiguous_on(int_identity_projection(), mr,
                                               copy, stream, device_data));
}

TEST_F(CudaSanityContiguousOn, FalseOrdered) {
    std::vector<int> host_vector;

    for (int i = 0; i < 5000; ++i) {
        if (i == 105) {
            host_vector.push_back(5);
        } else {
            for (int j = 0; j < i; ++j) {
                host_vector.push_back(i);
            }
        }
    }

    auto device_data = copy.to(vecmem::get_data(host_vector), mr,
                               vecmem::copy::type::host_to_device);

    ASSERT_FALSE(traccc::cuda::is_contiguous_on(int_identity_projection(), mr,
                                                copy, stream, device_data));
}

TEST_F(CudaSanityContiguousOn, FalseRandom) {
    std::vector<int> host_vector;

    for (int i : {603, 6432, 1, 3, 67, 1, 1111}) {
        for (int j = 0; j < i; ++j) {
            host_vector.push_back(i);
        }
    }

    auto device_data = copy.to(vecmem::get_data(host_vector), mr,
                               vecmem::copy::type::host_to_device);

    ASSERT_FALSE(traccc::cuda::is_contiguous_on(int_identity_projection(), mr,
                                                copy, stream, device_data));
}
