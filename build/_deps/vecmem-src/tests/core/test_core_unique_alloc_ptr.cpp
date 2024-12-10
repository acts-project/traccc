/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include <cstring>
#include <type_traits>

#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/memory/unique_ptr.hpp"

namespace {
struct NonTriviallyConstructibleType {
    int i = 0;
};
}  // namespace

class CoreUniqueAllocPtrTest : public testing::Test {
protected:
    vecmem::host_memory_resource mr;
};

TEST_F(CoreUniqueAllocPtrTest, EmptyPointer) {
    vecmem::unique_alloc_ptr<int> ptr;

    ASSERT_EQ(ptr, nullptr);
}

TEST_F(CoreUniqueAllocPtrTest, NullPointer) {
    vecmem::unique_alloc_ptr<int> ptr = nullptr;

    ASSERT_EQ(ptr, nullptr);
}

TEST_F(CoreUniqueAllocPtrTest, AllocSingle) {
    vecmem::unique_alloc_ptr<int> ptr = vecmem::make_unique_alloc<int>(mr);

    ASSERT_NE(ptr, nullptr);

    *ptr = 8;

    ASSERT_EQ(*ptr, 8);
}

TEST_F(CoreUniqueAllocPtrTest, AllocUnboundedArray) {
    vecmem::unique_alloc_ptr<int[]> ptr =
        vecmem::make_unique_alloc<int[]>(mr, 4);

    ASSERT_NE(ptr, nullptr);

    ptr[0] = 0;
    ptr[1] = 1;
    ptr[2] = 2;
    ptr[3] = 3;

    ASSERT_EQ(ptr[0], 0);
    ASSERT_EQ(ptr[1], 1);
    ASSERT_EQ(ptr[2], 2);
    ASSERT_EQ(ptr[3], 3);
}

TEST_F(CoreUniqueAllocPtrTest, AllocUnbounded2DArray) {
    vecmem::unique_alloc_ptr<int[][3]> ptr =
        vecmem::make_unique_alloc<int[][3]>(mr, 2);

    ASSERT_NE(ptr, nullptr);

    ptr[0][0] = 0;
    ptr[0][1] = 1;
    ptr[0][2] = 2;
    ptr[1][0] = 3;
    ptr[1][1] = 4;
    ptr[1][2] = 5;

    ASSERT_EQ(ptr[0][0], 0);
    ASSERT_EQ(ptr[0][1], 1);
    ASSERT_EQ(ptr[0][2], 2);
    ASSERT_EQ(ptr[1][0], 3);
    ASSERT_EQ(ptr[1][1], 4);
    ASSERT_EQ(ptr[1][2], 5);
}

TEST_F(CoreUniqueAllocPtrTest, MoveSingle) {
    vecmem::unique_alloc_ptr<int> ptr1 = vecmem::make_unique_alloc<int>(mr);
    vecmem::unique_alloc_ptr<int> ptr2;

    ASSERT_NE(ptr1, nullptr);
    ASSERT_EQ(ptr2, nullptr);

    *ptr1 = 8;

    ASSERT_EQ(*ptr1, 8);

    ptr2 = std::move(ptr1);

    ASSERT_EQ(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);

    ASSERT_EQ(*ptr2, 8);
}

TEST_F(CoreUniqueAllocPtrTest, MoveUnboundedArray) {
    vecmem::unique_alloc_ptr<int[]> ptr1 =
        vecmem::make_unique_alloc<int[]>(mr, 4);
    vecmem::unique_alloc_ptr<int[]> ptr2;

    ASSERT_NE(ptr1, nullptr);
    ASSERT_EQ(ptr2, nullptr);

    ptr1[0] = 8;

    ASSERT_EQ(ptr1[0], 8);

    ptr2 = std::move(ptr1);

    ASSERT_EQ(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);

    ASSERT_EQ(ptr2[0], 8);
}

TEST_F(CoreUniqueAllocPtrTest, DeallocateSingle) {
    vecmem::unique_alloc_ptr<int> ptr1 = vecmem::make_unique_alloc<int>(mr);

    ptr1 = nullptr;

    ASSERT_EQ(ptr1, nullptr);
}

TEST_F(CoreUniqueAllocPtrTest, CheckTypeTraits) {
    EXPECT_TRUE(std::is_trivially_copyable_v<NonTriviallyConstructibleType>);
    EXPECT_FALSE(std::is_trivially_default_constructible_v<
                 NonTriviallyConstructibleType>);
}

TEST_F(CoreUniqueAllocPtrTest, AllocateSingleByCopy) {
    int src = 18;

    vecmem::unique_alloc_ptr<int> ptr =
        vecmem::make_unique_alloc<int>(mr, &src, std::memcpy);

    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(*ptr, 18);
}

TEST_F(CoreUniqueAllocPtrTest, AllocateArrayByCopy) {
    int src[5] = {13, 61, 59, 42, 74};

    vecmem::unique_alloc_ptr<int[]> ptr =
        vecmem::make_unique_alloc<int[]>(mr, 5, src, std::memcpy);

    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(ptr[0], 13);
    EXPECT_EQ(ptr[1], 61);
    EXPECT_EQ(ptr[2], 59);
    EXPECT_EQ(ptr[3], 42);
    EXPECT_EQ(ptr[4], 74);
}

TEST_F(CoreUniqueAllocPtrTest, AllocateSingleNonConstructibleByCopy) {
    NonTriviallyConstructibleType src;

    src.i = 51;

    vecmem::unique_alloc_ptr<NonTriviallyConstructibleType> ptr =
        vecmem::make_unique_alloc<NonTriviallyConstructibleType>(mr, &src,
                                                                 std::memcpy);

    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(ptr->i, 51);
}

TEST_F(CoreUniqueAllocPtrTest, AllocateArrayNonConstructibleByCopy) {
    NonTriviallyConstructibleType src[5];

    src[0].i = 13;
    src[1].i = 61;
    src[2].i = 59;
    src[3].i = 42;
    src[4].i = 74;

    vecmem::unique_alloc_ptr<NonTriviallyConstructibleType[]> ptr =
        vecmem::make_unique_alloc<NonTriviallyConstructibleType[]>(mr, 5, src,
                                                                   std::memcpy);

    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(ptr[0].i, 13);
    EXPECT_EQ(ptr[1].i, 61);
    EXPECT_EQ(ptr[2].i, 59);
    EXPECT_EQ(ptr[3].i, 42);
    EXPECT_EQ(ptr[4].i, 74);
}
