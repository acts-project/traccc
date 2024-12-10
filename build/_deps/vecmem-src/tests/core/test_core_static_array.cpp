/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/containers/static_array.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

class core_static_array_test : public testing::Test {};

TEST_F(core_static_array_test, basic_write_read) {
    vecmem::static_array<int, 3> arr;

    arr[0] = 5;
    arr[1] = 11;
    arr[2] = 3;

    EXPECT_EQ(arr[0], 5);
    EXPECT_EQ(arr[1], 11);
    EXPECT_EQ(arr[2], 3);
}

TEST_F(core_static_array_test, bounds_check_exception) {
    vecmem::static_array<int, 3> arr;

    EXPECT_NO_THROW(arr.at(2));
    EXPECT_THROW(arr.at(3), std::out_of_range);
}

TEST_F(core_static_array_test, initializer) {
    vecmem::static_array<int, 3> arr{4, 76, 1};

    EXPECT_EQ(arr[0], 4);
    EXPECT_EQ(arr[1], 76);
    EXPECT_EQ(arr[2], 1);
}

TEST_F(core_static_array_test, bracket_initializer) {
    vecmem::static_array<int, 3> arr({4, 76, 1});

    EXPECT_EQ(arr[0], 4);
    EXPECT_EQ(arr[1], 76);
    EXPECT_EQ(arr[2], 1);
}

TEST_F(core_static_array_test, assignment) {
    vecmem::static_array<int, 3> proto{4, 76, 1};
    vecmem::static_array<int, 3> arr = proto;

    EXPECT_EQ(arr[0], 4);
    EXPECT_EQ(arr[1], 76);
    EXPECT_EQ(arr[2], 1);
}

TEST_F(core_static_array_test, copy) {
    vecmem::static_array<int, 3> proto{4, 76, 1};
    vecmem::static_array<int, 3> arr(proto);

    EXPECT_EQ(arr[0], 4);
    EXPECT_EQ(arr[1], 76);
    EXPECT_EQ(arr[2], 1);
}

TEST_F(core_static_array_test, equality) {
    vecmem::static_array<int, 3> a1{4, 76, 1};
    vecmem::static_array<int, 3> a2{4, 76, 1};
    vecmem::static_array<int, 3> a3{4, 76, 2};

    EXPECT_TRUE(a1 == a1);
    EXPECT_TRUE(a2 == a2);
    EXPECT_TRUE(a3 == a3);
    EXPECT_TRUE(a1 == a2);
    EXPECT_TRUE(a2 == a1);
    EXPECT_FALSE(a1 == a3);
    EXPECT_FALSE(a2 == a3);
    EXPECT_FALSE(a3 == a1);
    EXPECT_FALSE(a3 == a2);
}

TEST_F(core_static_array_test, inequality) {
    vecmem::static_array<int, 3> a1{4, 76, 1};
    vecmem::static_array<int, 3> a2{4, 76, 1};
    vecmem::static_array<int, 3> a3{4, 76, 2};

    EXPECT_FALSE(a1 != a1);
    EXPECT_FALSE(a2 != a2);
    EXPECT_FALSE(a3 != a3);
    EXPECT_FALSE(a1 != a2);
    EXPECT_FALSE(a2 != a1);
    EXPECT_TRUE(a1 != a3);
    EXPECT_TRUE(a2 != a3);
    EXPECT_TRUE(a3 != a1);
    EXPECT_TRUE(a3 != a2);
}

TEST_F(core_static_array_test, front) {
    vecmem::static_array<int, 3> arr1{4, 76, 1};
    vecmem::static_array<int, 1> arr2{7};

    EXPECT_EQ(arr1.front(), 4);
    EXPECT_EQ(arr2.front(), 7);
}

TEST_F(core_static_array_test, back) {
    vecmem::static_array<int, 3> arr1{4, 76, 1};
    vecmem::static_array<int, 1> arr2{7};

    EXPECT_EQ(arr1.back(), 1);
    EXPECT_EQ(arr2.back(), 7);
}

TEST_F(core_static_array_test, matrix) {
    vecmem::static_array<vecmem::static_array<int, 3>, 3> matrix;

    matrix[0] = {5, 1, 8};
    matrix[1] = {9, 2, 4};
    matrix[2] = {0, 3, 7};

    EXPECT_EQ(matrix[0][0], 5);
    EXPECT_EQ(matrix[0][2], 8);
    EXPECT_EQ(matrix[1][1], 2);
    EXPECT_EQ(matrix[2][0], 0);
    EXPECT_EQ(matrix[2][2], 7);
}

TEST_F(core_static_array_test, get) {

    constexpr vecmem::static_array<int, 3> a{5, 20, 22};
    constexpr int a0 = vecmem::get<0>(a);
    constexpr int a1 = vecmem::get<1>(a);
    EXPECT_EQ(a0, 5);
    EXPECT_EQ(a1, 20);
}
