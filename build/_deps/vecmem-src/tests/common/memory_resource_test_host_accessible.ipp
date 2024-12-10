/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <algorithm>
#include <vector>

/// Custom non-trivial type used in the tests.
struct TestType {
    TestType(int a, long b = 123) : m_a(a), m_b(b) {}
    int m_a;
    long m_b;
};
/// Helper operator for @c TestType
bool operator==(const TestType& value1, const TestType& value2) {
    return ((value1.m_a == value2.m_a) && (value1.m_b == value2.m_b));
}

/// Comparison operator for fundamental types
template <typename T>
bool almost_equal(const T& value1, const T& value2) {
    return (std::abs(value1 - value2) < 0.001);
}

/// Comparison operator for the custom test type
template <>
bool almost_equal<TestType>(const TestType& value1, const TestType& value2) {
    return (value1 == value2);
}

template <typename T>
void memory_resource_test_host_accessible::test_host_accessible_resource(
    vecmem::vector<T>& test_vector) {

    // Set up the test vector, and create a reference vector.
    std::vector<T> reference_vector;
    reference_vector.reserve(100);
    test_vector.reserve(100);

    // Fill them up with some dummy content.
    for (int i = 0; i < 20; ++i) {
        reference_vector.push_back(i * 2);
        test_vector.push_back(i * 2);
    }
    // Make sure that they are the same.
    EXPECT_EQ(reference_vector.size(), test_vector.size());
    EXPECT_TRUE(std::equal(reference_vector.begin(), reference_vector.end(),
                           test_vector.begin()));

    // Remove a couple of elements from the vectors.
    for (int i : {26, 38, 25}) {
        (void)std::remove(reference_vector.begin(), reference_vector.end(), i);
        (void)std::remove(test_vector.begin(), test_vector.end(), i);
    }
    // Make sure that they are still the same.
    EXPECT_EQ(reference_vector.size(), test_vector.size());
    EXPECT_TRUE(std::equal(reference_vector.begin(), reference_vector.end(),
                           test_vector.begin(), ::almost_equal<T>));
}

/// Test the host accessible memory resource with an integer type.
TEST_P(memory_resource_test_host_accessible, int_value) {
    vecmem::vector<int> test_vector(GetParam());
    test_host_accessible_resource(test_vector);
}

/// Test the host accessible memory resource with a floating point type.
TEST_P(memory_resource_test_host_accessible, double_value) {
    vecmem::vector<double> test_vector(GetParam());
    test_host_accessible_resource(test_vector);
}

/// Test the host accessible memory resource with a custom type.
TEST_P(memory_resource_test_host_accessible, custom_value) {
    vecmem::vector< ::TestType> test_vector(GetParam());
    test_host_accessible_resource(test_vector);
}
