/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/utils/tuple.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <type_traits>

TEST(core_tuple_test, set_get) {

    // Construct trivial tuples in a few different ways.
    vecmem::tuple<int, float, double> t1;
    (void)t1;
    vecmem::tuple<float, int> t2{2.f, 3};
    vecmem::tuple<double, int> t3{t2};  // Type mismatch on purpose!

    // Get/set elements in those tuples.
    EXPECT_FLOAT_EQ(vecmem::get<0>(t2), 2.f);
    EXPECT_EQ(vecmem::get<1>(t2), 3);

    vecmem::get<0>(t3) = 4.;
    vecmem::get<1>(t3) = 6;
    EXPECT_DOUBLE_EQ(vecmem::get<0>(t3), 4.f);
    EXPECT_EQ(vecmem::get<1>(t3), 6);
}

TEST(core_tuple_test, tie) {

    // Exercise vecmem::tie(...).
    int value1 = 0;
    float value2 = 1.f;
    double value3 = 2.;

    auto t = vecmem::tie(value1, value2, value3);
    EXPECT_EQ(vecmem::get<0>(t), 0);
    EXPECT_FLOAT_EQ(vecmem::get<1>(t), 1.f);
    EXPECT_DOUBLE_EQ(vecmem::get<2>(t), 2.);

    vecmem::get<0>(t) = 3;
    vecmem::get<1>(t) = 4.f;
    vecmem::get<2>(t) = 5.;
    EXPECT_EQ(value1, 3);
    EXPECT_FLOAT_EQ(value2, 4.f);
    EXPECT_DOUBLE_EQ(value3, 5.);
}

TEST(core_tuple_test, element) {

    // Exercise vecmem::tuple_element.
    static constexpr bool type_check1 = std::is_same_v<
        vecmem::tuple_element<1, vecmem::tuple<int, float, double>>::type,
        float>;
    EXPECT_TRUE(type_check1);
    static constexpr bool type_check2 = std::is_same_v<
        vecmem::tuple_element_t<2, vecmem::tuple<int, float, double>>, double>;
    EXPECT_TRUE(type_check2);
}

TEST(core_tuple_test, make) {

    // Exercise vecmem::make_tuple(...).
    auto t = vecmem::make_tuple(1, 2u, 3.f, 4.);
    EXPECT_EQ(vecmem::get<0>(t), 1);
    EXPECT_EQ(vecmem::get<1>(t), 2);
    EXPECT_FLOAT_EQ(vecmem::get<2>(t), 3.f);
    EXPECT_DOUBLE_EQ(vecmem::get<3>(t), 4.);

    static constexpr bool type_check1 =
        std::is_same_v<vecmem::tuple_element_t<0, decltype(t)>, int>;
    EXPECT_TRUE(type_check1);
    static constexpr bool type_check2 =
        std::is_same_v<vecmem::tuple_element_t<1, decltype(t)>, unsigned int>;
    EXPECT_TRUE(type_check2);
    static constexpr bool type_check3 =
        std::is_same_v<vecmem::tuple_element_t<2, decltype(t)>, float>;
    EXPECT_TRUE(type_check3);
    static constexpr bool type_check4 =
        std::is_same_v<vecmem::tuple_element_t<3, decltype(t)>, double>;
    EXPECT_TRUE(type_check4);
}
