/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/containers/array.hpp"
#include "vecmem/containers/device_array.hpp"
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/containers/static_array.hpp"
#include "vecmem/containers/static_vector.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/host_memory_resource.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <algorithm>
#include <numeric>

/// Test case for the custom container types
class core_container_test : public testing::Test {

protected:
    /// Memory resource to use in the tests
    vecmem::host_memory_resource m_resource;
    /// Test vector used for testing all of the custom containers
    vecmem::vector<int> m_reference_vector = {{1, 2, 5, 6, 3, 6, 1, 7, 9},
                                              &m_resource};

};  // class core_container_test

/// Test(s) for @c vecmem::const_device_vector
TEST_F(core_container_test, const_device_vector) {

    const vecmem::device_vector<const int> test_vector(
        vecmem::get_data(m_reference_vector));
    EXPECT_EQ(test_vector.size(), m_reference_vector.size());
    EXPECT_EQ(test_vector.empty(), m_reference_vector.empty());
    EXPECT_TRUE(std::equal(m_reference_vector.begin(), m_reference_vector.end(),
                           test_vector.begin()));
    EXPECT_EQ(std::accumulate(test_vector.begin(), test_vector.end(), 0),
              std::accumulate(test_vector.rbegin(), test_vector.rend(), 0));
    for (std::size_t i = 0; i < m_reference_vector.size(); ++i) {
        const vecmem::device_vector<const int>::size_type ii =
            static_cast<vecmem::device_vector<const int>::size_type>(i);
        EXPECT_EQ(test_vector.at(ii), m_reference_vector.at(i));
        EXPECT_EQ(test_vector[ii], m_reference_vector[i]);
    }
}

/// Test(s) for @c vecmem::device_vector
TEST_F(core_container_test, device_vector) {

    const vecmem::device_vector<int> test_vector(
        vecmem::get_data(m_reference_vector));
    EXPECT_EQ(test_vector.size(), m_reference_vector.size());
    EXPECT_EQ(test_vector.empty(), m_reference_vector.empty());
    EXPECT_TRUE(std::equal(m_reference_vector.begin(), m_reference_vector.end(),
                           test_vector.begin()));
    EXPECT_EQ(std::accumulate(test_vector.begin(), test_vector.end(), 0),
              std::accumulate(test_vector.rbegin(), test_vector.rend(), 0));
    for (std::size_t i = 0; i < m_reference_vector.size(); ++i) {
        const vecmem::device_vector<int>::size_type ii =
            static_cast<vecmem::device_vector<int>::size_type>(i);
        EXPECT_EQ(test_vector.at(ii), m_reference_vector.at(i));
        EXPECT_EQ(test_vector[ii], m_reference_vector[i]);
    }
}

/// Test(s) for @c vecmem::static_vector
TEST_F(core_container_test, static_vector) {

    vecmem::static_vector<int, 20> test_vector(m_reference_vector.size());
    std::copy(m_reference_vector.begin(), m_reference_vector.end(),
              test_vector.begin());
    EXPECT_EQ(test_vector.size(), m_reference_vector.size());
    EXPECT_TRUE(std::equal(m_reference_vector.begin(), m_reference_vector.end(),
                           test_vector.begin()));
}

/// Test(s) for @c vecmem::array
TEST_F(core_container_test, array) {

    vecmem::array<int, 20> test_array(m_resource);
    std::copy(m_reference_vector.begin(), m_reference_vector.end(),
              test_array.begin());
    EXPECT_TRUE(std::equal(m_reference_vector.begin(), m_reference_vector.end(),
                           test_array.begin()));
}

/// Test(s) for @c vecmem::const_device_array
TEST_F(core_container_test, const_device_array) {

    const vecmem::device_array<const int, 9> test_array(
        vecmem::get_data(m_reference_vector));
    EXPECT_EQ(test_array.size(), m_reference_vector.size());
    EXPECT_EQ(test_array.empty(), m_reference_vector.empty());
    EXPECT_TRUE(std::equal(m_reference_vector.begin(), m_reference_vector.end(),
                           test_array.begin()));
    EXPECT_EQ(std::accumulate(test_array.begin(), test_array.end(), 0),
              std::accumulate(test_array.rbegin(), test_array.rend(), 0));
    for (std::size_t i = 0; i < m_reference_vector.size(); ++i) {
        EXPECT_EQ(test_array.at(i), m_reference_vector.at(i));
        EXPECT_EQ(test_array[i], m_reference_vector[i]);
    }
}

/// Test(s) for @c vecmem::device_array
TEST_F(core_container_test, device_array) {

    const vecmem::device_array<int, 9> test_array(
        vecmem::get_data(m_reference_vector));
    EXPECT_EQ(test_array.size(), m_reference_vector.size());
    EXPECT_EQ(test_array.empty(), m_reference_vector.empty());
    EXPECT_TRUE(std::equal(m_reference_vector.begin(), m_reference_vector.end(),
                           test_array.begin()));
    EXPECT_EQ(std::accumulate(test_array.begin(), test_array.end(), 0),
              std::accumulate(test_array.rbegin(), test_array.rend(), 0));
    for (std::size_t i = 0; i < m_reference_vector.size(); ++i) {
        EXPECT_EQ(test_array.at(i), m_reference_vector.at(i));
        EXPECT_EQ(test_array[i], m_reference_vector[i]);
    }
}

/// Test(s) for @c vecmem::static_array
TEST_F(core_container_test, static_array) {

    static constexpr int ARRAY_SIZE = 10;
    const vecmem::static_array<int, ARRAY_SIZE> test_array1 = {0, 1, 2, 3, 4,
                                                               5, 6, 7, 8, 9};
    EXPECT_EQ(test_array1.size(), ARRAY_SIZE);
    EXPECT_EQ(test_array1.max_size(), ARRAY_SIZE);
    EXPECT_FALSE(test_array1.empty());
    const vecmem::static_array<int, ARRAY_SIZE> test_array2 = test_array1;
    vecmem::static_array<int, ARRAY_SIZE> test_array3 = {2, 3, 4, 5, 6,
                                                         7, 8, 9, 0, 1};
    EXPECT_EQ(test_array1, test_array2);
    EXPECT_NE(test_array1, test_array3);
    EXPECT_TRUE(std::equal(test_array1.begin(), test_array1.end(),
                           test_array2.begin()));
    EXPECT_FALSE(std::equal(test_array1.begin(), test_array1.end(),
                            test_array3.begin()));
    EXPECT_EQ(std::accumulate(test_array1.begin(), test_array1.end(), 0),
              std::accumulate(test_array2.rbegin(), test_array2.rend(), 0));
    test_array3.fill(12);
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        EXPECT_EQ(test_array3.at(static_cast<std::size_t>(i)), 12);
    }
    const vecmem::static_array<int, 0> test_array4{};
    EXPECT_EQ(test_array4.size(), 0);
    EXPECT_EQ(test_array4.max_size(), 0);
    EXPECT_TRUE(test_array4.empty());

    constexpr vecmem::static_array<int, 0> test_array5{};
    constexpr auto test_array5_size = test_array5.size();
    EXPECT_EQ(test_array5_size, 0);
    constexpr auto test_array5_max_size = test_array5.max_size();
    EXPECT_EQ(test_array5_max_size, 0);
    constexpr bool test_array5_empty = test_array5.empty();
    EXPECT_TRUE(test_array5_empty);

    constexpr vecmem::static_array<int, ARRAY_SIZE> test_array6 = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    constexpr auto test_array6_size = test_array6.size();
    EXPECT_EQ(test_array6_size, ARRAY_SIZE);
    constexpr auto test_array6_max_size = test_array6.max_size();
    EXPECT_EQ(test_array6_max_size, ARRAY_SIZE);
    constexpr bool test_array6_empty = test_array6.empty();
    EXPECT_FALSE(test_array6_empty);
}
