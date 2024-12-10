/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <gtest/gtest.h>

#include <covfie/core/utility/numeric.hpp>

TEST(TestRoundPow2, Input1)
{
    EXPECT_EQ(covfie::utility::round_pow2(1), 1);
}

TEST(TestRoundPow2, Input2)
{
    EXPECT_EQ(covfie::utility::round_pow2(2), 2);
}

TEST(TestRoundPow2, Input3)
{
    EXPECT_EQ(covfie::utility::round_pow2(3), 4);
}

TEST(TestRoundPow2, Input4)
{
    EXPECT_EQ(covfie::utility::round_pow2(4), 4);
}

TEST(TestRoundPow2, Input511)
{
    EXPECT_EQ(covfie::utility::round_pow2(511), 512);
}

TEST(TestRoundPow2, Input512)
{
    EXPECT_EQ(covfie::utility::round_pow2(512), 512);
}

TEST(TestRoundPow2, Input513)
{
    EXPECT_EQ(covfie::utility::round_pow2(513), 1024);
}

TEST(TestIPow, PowerZero1)
{
    EXPECT_EQ(covfie::utility::ipow(0, 0), 1);
}

TEST(TestIPow, PowerZero2)
{
    EXPECT_EQ(covfie::utility::ipow(0, 5), 0);
}

TEST(TestIPow, PowerOne1)
{
    EXPECT_EQ(covfie::utility::ipow(1, 0), 1);
}

TEST(TestIPow, PowerOne2)
{
    EXPECT_EQ(covfie::utility::ipow(1, 1), 1);
}

TEST(TestIPow, PowerOne3)
{
    EXPECT_EQ(covfie::utility::ipow(1, 5), 1);
}

TEST(TestIPow, PowerExample1)
{
    EXPECT_EQ(covfie::utility::ipow(5, 2), 25);
}

TEST(TestIPow, PowerExample2)
{
    EXPECT_EQ(covfie::utility::ipow(8, 7), 2097152);
}
