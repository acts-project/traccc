/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <gtest/gtest.h>

#include <covfie/core/utility/nd_map.hpp>

TEST(TestNDMap, Simple1DAdd)
{
    std::size_t i = 0;

    covfie::utility::nd_map<covfie::array::array<std::size_t, 1>>(
        std::function([&i](covfie::array::array<std::size_t, 1> t) {
            i += t.at(0);
        }),
        {10u}
    );

    EXPECT_EQ(i, 45);
}

TEST(TestNDMap, Simple2DAdd)
{
    std::size_t i = 0;

    covfie::utility::nd_map<covfie::array::array<std::size_t, 2>>(
        std::function([&i](covfie::array::array<std::size_t, 2> t) {
            i += (t.at(0) + t.at(1));
        }),
        {5u, 10u}
    );

    EXPECT_EQ(i, 325);
}
