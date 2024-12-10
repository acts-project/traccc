/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <type_traits>
#include <vector>

#include "vecmem/memory/allocator.hpp"
#include "vecmem/memory/host_memory_resource.hpp"

class test_class {
public:
    test_class() : m_int_2(11), m_bool_2(true) {}
    test_class(int n) : m_int_2(n), m_bool_2(false) {}
    test_class(int n, int m) : m_int_2(n), m_bool_2(m > 100) {}

    int m_int_1 = 5;
    int m_int_2;

    bool m_bool_1 = false;
    bool m_bool_2;
};

class core_allocator_test : public testing::Test {
protected:
    vecmem::host_memory_resource m_upstream;
    vecmem::allocator m_alloc{m_upstream};
};

TEST_F(core_allocator_test, basic) {
    void* p = m_alloc.allocate_bytes(1024);

    EXPECT_NE(p, nullptr);

    m_alloc.deallocate_bytes(p, 1024);
}

TEST_F(core_allocator_test, primitive) {
    int* p = m_alloc.allocate_object<int>();

    EXPECT_NE(p, nullptr);

    *p = 5;

    EXPECT_EQ(*p, 5);

    m_alloc.deallocate_object<int>(p);
}

TEST_F(core_allocator_test, array) {
    int* p = m_alloc.allocate_object<int>(10);

    EXPECT_NE(p, nullptr);

    for (int i = 0; i < 10; ++i) {
        p[i] = i;
    }

    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(p[i], i);
    }

    m_alloc.deallocate_object<int>(p, 10);
}

TEST_F(core_allocator_test, constructor) {
    test_class* p1 = m_alloc.new_object<test_class>();
    test_class* p2 = m_alloc.new_object<test_class>(12);
    test_class* p3 = m_alloc.new_object<test_class>(21, 611);
    test_class* p4 = m_alloc.new_object<test_class>(21, 15);

    EXPECT_NE(p1, nullptr);
    EXPECT_NE(p2, nullptr);
    EXPECT_NE(p3, nullptr);
    EXPECT_NE(p4, nullptr);

    EXPECT_EQ(p1->m_int_1, 5);
    EXPECT_EQ(p1->m_int_2, 11);
    EXPECT_EQ(p1->m_bool_1, false);
    EXPECT_EQ(p1->m_bool_2, true);

    EXPECT_EQ(p2->m_int_1, 5);
    EXPECT_EQ(p2->m_int_2, 12);
    EXPECT_EQ(p2->m_bool_1, false);
    EXPECT_EQ(p2->m_bool_2, false);

    EXPECT_EQ(p3->m_int_1, 5);
    EXPECT_EQ(p3->m_int_2, 21);
    EXPECT_EQ(p3->m_bool_1, false);
    EXPECT_EQ(p3->m_bool_2, true);

    EXPECT_EQ(p4->m_int_1, 5);
    EXPECT_EQ(p4->m_int_2, 21);
    EXPECT_EQ(p4->m_bool_1, false);
    EXPECT_EQ(p4->m_bool_2, false);

    m_alloc.delete_object(p1);
    m_alloc.delete_object(p2);
    m_alloc.delete_object(p3);
    m_alloc.delete_object(p4);
}
