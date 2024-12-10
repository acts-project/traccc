/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "detray/utils/type_list.hpp"

// Google Test include(s).
#include <gtest/gtest.h>

// Test type list implementation
GTEST_TEST(detray_utils, type_list) {
    using namespace detray;

    using list = types::list<float, int, double>;

    static_assert(std::is_same_v<types::front<list>, float>,
                  "Could not access type list front");

    static_assert(std::is_same_v<types::back<list>, double>,
                  "Could not access type list back");

    static_assert(std::is_same_v<types::push_back<list, char>,
                                 types::list<float, int, double, char>>,
                  "Failed to push back new type");

    static_assert(std::is_same_v<types::push_front<list, char>,
                                 types::list<char, float, int, double>>,
                  "Failed to push front new type");

    static_assert(types::size<list> == 3ul, "Incorrect size");

    static_assert(std::is_same_v<types::at<list, 1>, int>,
                  "Failed access type");

    types::print<list>();
}
