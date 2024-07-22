/*
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// vecmem includes
#include <vecmem/memory/host_memory_resource.hpp>

// traccc includes
#include <traccc/definitions/qualifiers.hpp>
#include <traccc/sanity/ordered_on.hpp>

// GTest include(s).
#include <gtest/gtest.h>

struct int_lt_relation {
    TRACCC_HOST_DEVICE
    bool operator()(const int& a, const int& b) const { return a < b; }
};

struct int_leq_relation {
    TRACCC_HOST_DEVICE
    bool operator()(const int& a, const int& b) const { return a <= b; }
};

class CPUSanityOrderedOn : public testing::Test {
    protected:
    CPUSanityOrderedOn() {}
};

TEST_F(CPUSanityOrderedOn, TrueConsecutiveNoRepeatsLeq) {
    std::vector<int> host_vector;

    for (int i = 0; i < 500000; ++i) {
        host_vector.push_back(i);
    }

    auto device_data = vecmem::get_data(host_vector);

    ASSERT_TRUE(traccc::host::is_ordered_on(int_leq_relation(), device_data));
}

TEST_F(CPUSanityOrderedOn, TrueConsecutiveNoRepeatsLt) {
    std::vector<int> host_vector;

    for (int i = 0; i < 500000; ++i) {
        host_vector.push_back(i);
    }

    auto device_data = vecmem::get_data(host_vector);

    ASSERT_TRUE(traccc::host::is_ordered_on(int_lt_relation(), device_data));
}

TEST_F(CPUSanityOrderedOn, TrueConsecutiveRepeatsLeq) {
    std::vector<int> host_vector;

    for (int i = 0; i < 5000; ++i) {
        for (int j = 0; j < i; ++j) {
            host_vector.push_back(i);
        }
    }

    auto device_data = vecmem::get_data(host_vector);

    ASSERT_TRUE(traccc::host::is_ordered_on(int_leq_relation(), device_data));
}

TEST_F(CPUSanityOrderedOn, FalseConsecutiveRepeatLt) {
    std::vector<int> host_vector;

    for (int i = 0; i < 5000; ++i) {
        for (int j = 0; j < i; ++j) {
            host_vector.push_back(i);
        }
    }

    auto device_data = vecmem::get_data(host_vector);

    ASSERT_FALSE(traccc::host::is_ordered_on(int_lt_relation(), device_data));
}

TEST_F(CPUSanityOrderedOn, TrueConsecutivePathologicalFirstLeq) {
    std::vector<int> host_vector;

    host_vector.push_back(4000);

    for (int i = 0; i < 5000; ++i) {
        for (int j = 0; j < i; ++j) {
            host_vector.push_back(i);
        }
    }

    auto device_data = vecmem::get_data(host_vector);

    ASSERT_FALSE(traccc::host::is_ordered_on(int_leq_relation(), device_data));
}

TEST_F(CPUSanityOrderedOn, TrueConsecutivePathologicalLastLeq) {
    std::vector<int> host_vector;

    host_vector.push_back(2000);

    for (int i = 0; i < 5000; ++i) {
        for (int j = 0; j < i; ++j) {
            host_vector.push_back(i);
        }
    }

    auto device_data = vecmem::get_data(host_vector);

    ASSERT_FALSE(traccc::host::is_ordered_on(int_leq_relation(), device_data));
}
