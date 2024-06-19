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
#include <traccc/cuda/sanity/ordered_on.cuh>
#include <traccc/definitions/qualifiers.hpp>

// GTest include(s).
#include <gtest/gtest.h>

struct int_lt_relation {
    TRACCC_HOST_DEVICE
    bool operator()(const int& a, const int& b) { return a < b; }
};

struct int_leq_relation {
    TRACCC_HOST_DEVICE
    bool operator()(const int& a, const int& b) { return a <= b; }
};

class CudaSanityOrderedOn : public testing::Test {
    protected:
    CudaSanityOrderedOn() : copy(stream.cudaStream()) {}

    vecmem::cuda::device_memory_resource mr;
    traccc::cuda::stream stream;
    vecmem::cuda::async_copy copy;
};

TEST_F(CudaSanityOrderedOn, TrueConsecutiveNoRepeatsLeq) {
    std::vector<int> host_vector;

    for (int i = 0; i < 500000; ++i) {
        host_vector.push_back(i);
    }

    auto device_data = copy.to(vecmem::get_data(host_vector), mr,
                               vecmem::copy::type::host_to_device);

    ASSERT_TRUE(traccc::cuda::is_ordered_on(int_leq_relation(), mr, copy,
                                            stream, device_data));
}

TEST_F(CudaSanityOrderedOn, TrueConsecutiveNoRepeatsLt) {
    std::vector<int> host_vector;

    for (int i = 0; i < 500000; ++i) {
        host_vector.push_back(i);
    }

    auto device_data = copy.to(vecmem::get_data(host_vector), mr,
                               vecmem::copy::type::host_to_device);

    ASSERT_TRUE(traccc::cuda::is_ordered_on(int_lt_relation(), mr, copy, stream,
                                            device_data));
}

TEST_F(CudaSanityOrderedOn, TrueConsecutiveRepeatsLeq) {
    std::vector<int> host_vector;

    for (int i = 0; i < 5000; ++i) {
        for (int j = 0; j < i; ++j) {
            host_vector.push_back(i);
        }
    }

    auto device_data = copy.to(vecmem::get_data(host_vector), mr,
                               vecmem::copy::type::host_to_device);

    ASSERT_TRUE(traccc::cuda::is_ordered_on(int_leq_relation(), mr, copy,
                                            stream, device_data));
}

TEST_F(CudaSanityOrderedOn, FalseConsecutiveRepeatLt) {
    std::vector<int> host_vector;

    for (int i = 0; i < 5000; ++i) {
        for (int j = 0; j < i; ++j) {
            host_vector.push_back(i);
        }
    }

    auto device_data = copy.to(vecmem::get_data(host_vector), mr,
                               vecmem::copy::type::host_to_device);

    ASSERT_FALSE(traccc::cuda::is_ordered_on(int_lt_relation(), mr, copy,
                                             stream, device_data));
}
