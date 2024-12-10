/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/memory/unique_ptr.hpp"

class CoreUniqueObjPtrTest : public testing::Test {
protected:
    inline static std::size_t n_construct;
    inline static std::size_t n_destruct;

    class A {
    public:
        A(void) { ++n_construct; }

        ~A() { ++n_destruct; }
    };

    class B {
    public:
        B(void) = delete;

        B(int, float) { ++n_construct; }

        ~B() { ++n_destruct; }
    };

    vecmem::host_memory_resource mr;
};

TEST_F(CoreUniqueObjPtrTest, EmptyPointerPrimitive) {
    vecmem::unique_obj_ptr<int> ptr;

    ASSERT_EQ(ptr, nullptr);
}

TEST_F(CoreUniqueObjPtrTest, EmptyPointerComplex) {
    vecmem::unique_obj_ptr<A> ptr;

    ASSERT_EQ(ptr, nullptr);
}

TEST_F(CoreUniqueObjPtrTest, NullPointerPrimitive) {
    vecmem::unique_obj_ptr<int> ptr = nullptr;

    ASSERT_EQ(ptr, nullptr);
}

TEST_F(CoreUniqueObjPtrTest, NullPointerComplex) {
    vecmem::unique_obj_ptr<A> ptr = nullptr;

    ASSERT_EQ(ptr, nullptr);
}

TEST_F(CoreUniqueObjPtrTest, AllocSingleA) {
    std::size_t pre = n_construct;

    vecmem::unique_obj_ptr<A> ptr = vecmem::make_unique_obj<A>(mr);

    ASSERT_EQ(pre + 1, n_construct);
}

TEST_F(CoreUniqueObjPtrTest, AllocUnboundedArrayA) {
    std::size_t pre = n_construct;

    vecmem::unique_obj_ptr<A[]> ptr = vecmem::make_unique_obj<A[]>(mr, 5);

    ASSERT_EQ(pre + 5, n_construct);
}

TEST_F(CoreUniqueObjPtrTest, MoveSingleA) {
    std::size_t pre = n_construct;

    vecmem::unique_obj_ptr<A> ptr1 = vecmem::make_unique_obj<A>(mr);
    vecmem::unique_obj_ptr<A> ptr2;

    ASSERT_NE(ptr1, nullptr);
    ASSERT_EQ(ptr2, nullptr);

    ASSERT_EQ(pre + 1, n_construct);

    ptr2 = std::move(ptr1);

    ASSERT_EQ(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);

    ASSERT_EQ(pre + 1, n_construct);
}

TEST_F(CoreUniqueObjPtrTest, MoveUnboundedArrayA) {
    std::size_t pre = n_construct;

    vecmem::unique_obj_ptr<A[]> ptr1 = vecmem::make_unique_obj<A[]>(mr, 5);
    vecmem::unique_obj_ptr<A[]> ptr2;

    ASSERT_NE(ptr1, nullptr);
    ASSERT_EQ(ptr2, nullptr);

    ASSERT_EQ(pre + 5, n_construct);

    ptr2 = std::move(ptr1);

    ASSERT_EQ(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);

    ASSERT_EQ(pre + 5, n_construct);
}

TEST_F(CoreUniqueObjPtrTest, AllocSingleB) {
    std::size_t pre = n_construct;

    vecmem::unique_obj_ptr<B> ptr = vecmem::make_unique_obj<B>(mr, 5, 7.f);

    ASSERT_EQ(pre + 1, n_construct);
}

TEST_F(CoreUniqueObjPtrTest, DeallocSingleA) {
    std::size_t pre = n_destruct;

    vecmem::unique_obj_ptr<A> ptr1 = vecmem::make_unique_obj<A>(mr);
    vecmem::unique_obj_ptr<A> ptr2 = vecmem::make_unique_obj<A>(mr);

    ptr1 = nullptr;

    ASSERT_EQ(pre + 1, n_destruct);

    ptr2 = nullptr;

    ASSERT_EQ(pre + 2, n_destruct);
}

TEST_F(CoreUniqueObjPtrTest, DeallocUnboundedArrayA) {
    std::size_t pre = n_destruct;

    vecmem::unique_obj_ptr<A[]> ptr1 = vecmem::make_unique_obj<A[]>(mr, 12);
    vecmem::unique_obj_ptr<A[]> ptr2 = vecmem::make_unique_obj<A[]>(mr, 7);

    ptr1 = nullptr;

    ASSERT_EQ(pre + 12, n_destruct);

    ptr2 = nullptr;

    ASSERT_EQ(pre + 19, n_destruct);
}
