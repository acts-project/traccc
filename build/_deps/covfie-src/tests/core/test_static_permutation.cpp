/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <type_traits>

#include <gtest/gtest.h>

#include <covfie/core/utility/static_permutation.hpp>

TEST(TestStaticPermutation, SortEmpty)
{
    static constexpr bool same = std::is_same_v<
        covfie::utility::sort_index_sequence<std::index_sequence<>>::type,
        std::index_sequence<>>;
    EXPECT_TRUE(same);
}

TEST(TestStaticPermutation, SortSingle)
{
    static constexpr bool same = std::is_same_v<
        covfie::utility::sort_index_sequence<std::index_sequence<5>>::type,
        std::index_sequence<5>>;
    EXPECT_TRUE(same);
}

TEST(TestStaticPermutation, SortDouble)
{
    static constexpr bool same = std::is_same_v<
        covfie::utility::sort_index_sequence<std::index_sequence<5, 7>>::type,
        std::index_sequence<5, 7>>;
    EXPECT_TRUE(same);
}

TEST(TestStaticPermutation, SortMany)
{
    static constexpr bool same = std::is_same_v<
        covfie::utility::sort_index_sequence<
            std::index_sequence<1, 2, 0, 7, 11, 3>>::type,
        std::index_sequence<0, 1, 2, 3, 7, 11>>;
    EXPECT_TRUE(same);
}

TEST(TestStaticPermutation, SortManyDuplicates)
{
    static constexpr bool same = std::is_same_v<
        covfie::utility::sort_index_sequence<
            std::index_sequence<11, 0, 1, 2, 0, 7, 2, 11, 3, 1>>::type,
        std::index_sequence<0, 0, 1, 1, 2, 2, 3, 7, 11, 11>>;
    EXPECT_TRUE(same);
}

TEST(TestStaticPermutation, PermutationEmptyEmpty)
{
    static constexpr bool perm = covfie::utility::
        is_permutation<std::index_sequence<>, std::index_sequence<>>::value;
    EXPECT_TRUE(perm);
}

TEST(TestStaticPermutation, PermutationEmptySingle)
{
    static constexpr bool perm = covfie::utility::
        is_permutation<std::index_sequence<>, std::index_sequence<5>>::value;
    EXPECT_FALSE(perm);
}

TEST(TestStaticPermutation, PermutationSingleEmpty)
{
    static constexpr bool perm = covfie::utility::
        is_permutation<std::index_sequence<5>, std::index_sequence<>>::value;
    EXPECT_FALSE(perm);
}

TEST(TestStaticPermutation, PermutationSingleSingleTrue)
{
    static constexpr bool perm = covfie::utility::
        is_permutation<std::index_sequence<5>, std::index_sequence<5>>::value;
    EXPECT_TRUE(perm);
}

TEST(TestStaticPermutation, PermutationSingleSingleFalse)
{
    static constexpr bool perm = covfie::utility::
        is_permutation<std::index_sequence<5>, std::index_sequence<7>>::value;
    EXPECT_FALSE(perm);
}

TEST(TestStaticPermutation, PermutationManyManyTrue1)
{
    static constexpr bool perm = covfie::utility::is_permutation<
        std::index_sequence<2, 3, 1, 0>,
        std::index_sequence<0, 1, 2, 3>>::value;
    EXPECT_TRUE(perm);
}

TEST(TestStaticPermutation, PermutationManyManyTrue2)
{
    static constexpr bool perm = covfie::utility::is_permutation<
        std::index_sequence<7, 4, 2>,
        std::index_sequence<4, 7, 2>>::value;
    EXPECT_TRUE(perm);
}

TEST(TestStaticPermutation, PermutationManyManyFalse1)
{
    static constexpr bool perm = covfie::utility::is_permutation<
        std::index_sequence<7, 4, 2>,
        std::index_sequence<1, 7, 2>>::value;
    EXPECT_FALSE(perm);
}

TEST(TestStaticPermutation, PermutationManyManyFalse2)
{
    static constexpr bool perm = covfie::utility::is_permutation<
        std::index_sequence<7, 4, 2>,
        std::index_sequence<4, 7, 2, 4>>::value;
    EXPECT_FALSE(perm);
}
