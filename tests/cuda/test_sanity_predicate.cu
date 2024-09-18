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
#include <traccc/definitions/qualifiers.hpp>

#include "../../device/cuda/src/sanity/predicate.cuh"

// GTest include(s).
#include <gtest/gtest.h>

struct IsLessThan {
    __device__ bool operator()(int i) const { return i < m_val; }

    int m_val;
};

class CUDASanityPredicate : public testing::Test {
    protected:
    CUDASanityPredicate() : copy(stream.cudaStream()) {}

    vecmem::cuda::device_memory_resource mr;
    traccc::cuda::stream stream;
    vecmem::cuda::async_copy copy;
};

TEST_F(CUDASanityPredicate, TrueForAllTrue) {
    std::vector<int> host_vector;

    for (int i = 0; i < 5000; ++i) {
        host_vector.push_back(i);
    }

    auto device_data = copy.to(vecmem::get_data(host_vector), mr,
                               vecmem::copy::type::host_to_device);

    ASSERT_TRUE(traccc::cuda::true_for_all(IsLessThan{5001}, mr, copy, stream,
                                           device_data));
}

TEST_F(CUDASanityPredicate, TrueForAllFalse) {
    std::vector<int> host_vector;

    for (int i = 0; i < 5000; ++i) {
        host_vector.push_back(i);
    }

    auto device_data = copy.to(vecmem::get_data(host_vector), mr,
                               vecmem::copy::type::host_to_device);

    ASSERT_FALSE(traccc::cuda::true_for_all(IsLessThan{4500}, mr, copy, stream,
                                            device_data));
}

TEST_F(CUDASanityPredicate, TrueForAnyTrue) {
    std::vector<int> host_vector;

    for (int i = 0; i < 5000; ++i) {
        host_vector.push_back(i);
    }

    auto device_data = copy.to(vecmem::get_data(host_vector), mr,
                               vecmem::copy::type::host_to_device);

    ASSERT_TRUE(traccc::cuda::true_for_any(IsLessThan{1}, mr, copy, stream,
                                           device_data));
}

TEST_F(CUDASanityPredicate, TrueForAnyFalse) {
    std::vector<int> host_vector;

    for (int i = 0; i < 5000; ++i) {
        host_vector.push_back(i);
    }

    auto device_data = copy.to(vecmem::get_data(host_vector), mr,
                               vecmem::copy::type::host_to_device);

    ASSERT_FALSE(traccc::cuda::true_for_any(IsLessThan{0}, mr, copy, stream,
                                            device_data));
}

TEST_F(CUDASanityPredicate, FalseForAllTrue) {
    std::vector<int> host_vector;

    for (int i = 0; i < 5000; ++i) {
        host_vector.push_back(i);
    }

    auto device_data = copy.to(vecmem::get_data(host_vector), mr,
                               vecmem::copy::type::host_to_device);

    ASSERT_TRUE(traccc::cuda::false_for_all(IsLessThan{0}, mr, copy, stream,
                                            device_data));
}

TEST_F(CUDASanityPredicate, FalseForAllFalse) {
    std::vector<int> host_vector;

    for (int i = 0; i < 5000; ++i) {
        host_vector.push_back(i);
    }

    auto device_data = copy.to(vecmem::get_data(host_vector), mr,
                               vecmem::copy::type::host_to_device);

    ASSERT_FALSE(traccc::cuda::false_for_all(IsLessThan{1}, mr, copy, stream,
                                             device_data));
}

TEST_F(CUDASanityPredicate, FalseForAnyTrue) {
    std::vector<int> host_vector;

    for (int i = 0; i < 5000; ++i) {
        host_vector.push_back(i);
    }

    auto device_data = copy.to(vecmem::get_data(host_vector), mr,
                               vecmem::copy::type::host_to_device);

    ASSERT_TRUE(traccc::cuda::false_for_any(IsLessThan{1}, mr, copy, stream,
                                            device_data));
}

TEST_F(CUDASanityPredicate, FalseForAnyFalse) {
    std::vector<int> host_vector;

    for (int i = 0; i < 5000; ++i) {
        host_vector.push_back(i);
    }

    auto device_data = copy.to(vecmem::get_data(host_vector), mr,
                               vecmem::copy::type::host_to_device);

    ASSERT_FALSE(traccc::cuda::false_for_any(IsLessThan{6000}, mr, copy, stream,
                                             device_data));
}
