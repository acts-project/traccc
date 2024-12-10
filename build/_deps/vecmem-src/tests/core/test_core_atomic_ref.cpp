/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/device_atomic_ref.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

TEST(core_device_atomic_test, atomic_operator_equals) {
    int i = 0;
    vecmem::device_atomic_ref<int> a(i);
    a = 5;
    ASSERT_EQ(a.load(), 5);
}

TEST(core_device_atomic_test, atomic_load) {
    int i = 17;
    vecmem::device_atomic_ref<int> a(i);
    ASSERT_EQ(a.load(), 17);
}

TEST(core_device_atomic_test, atomic_compare_exchange_strong) {
    int i = 0, zero = 0, five = 5;
    vecmem::device_atomic_ref<int> a(i);
    ASSERT_TRUE(a.compare_exchange_strong(zero, 5));
    ASSERT_EQ(zero, 0);
    ASSERT_FALSE(a.compare_exchange_strong(zero, 5));
    ASSERT_EQ(zero, 5);
    ASSERT_TRUE(a.compare_exchange_strong(five, 0));
    ASSERT_EQ(five, 5);
    ASSERT_FALSE(a.compare_exchange_strong(five, 0));
    ASSERT_EQ(five, 0);
}

TEST(core_device_atomic_test, atomic_store) {
    int i = 0;
    vecmem::device_atomic_ref<int> a(i);
    ASSERT_EQ(a.load(), 0);
    a.store(5);
    ASSERT_EQ(a.load(), 5);
}

TEST(core_device_atomic_test, atomic_exchange) {
    int i = 0;
    vecmem::device_atomic_ref<int> a(i);
    ASSERT_EQ(a.load(), 0);
    ASSERT_EQ(a.exchange(5), 0);
    ASSERT_EQ(a.load(), 5);
    ASSERT_EQ(a.exchange(3), 5);
    ASSERT_EQ(a.load(), 3);
}

TEST(core_device_atomic_test, atomic_fetch_add) {
    int i = 0;
    vecmem::device_atomic_ref<int> a(i);
    ASSERT_EQ(a.load(), 0);
    ASSERT_EQ(a.fetch_add(5), 0);
    ASSERT_EQ(a.load(), 5);
    ASSERT_EQ(a.fetch_add(3), 5);
    ASSERT_EQ(a.load(), 8);
}

TEST(core_device_atomic_test, atomic_fetch_sub) {
    int i = 0;
    vecmem::device_atomic_ref<int> a(i);
    ASSERT_EQ(a.load(), 0);
    ASSERT_EQ(a.fetch_sub(5), 0);
    ASSERT_EQ(a.load(), -5);
    ASSERT_EQ(a.fetch_sub(3), -5);
    ASSERT_EQ(a.load(), -8);
}

TEST(core_device_atomic_test, atomic_fetch_and) {
    int i = 0b0101;
    vecmem::device_atomic_ref<int> a(i);
    ASSERT_EQ(a.load(), 0b0101);
    ASSERT_EQ(a.fetch_and(0b1100), 0b0101);
    ASSERT_EQ(a.load(), 0b0100);
    ASSERT_EQ(a.fetch_and(0b0000), 0b0100);
    ASSERT_EQ(a.load(), 0b0000);
}

TEST(core_device_atomic_test, atomic_fetch_or) {
    int i = 0b0101;
    vecmem::device_atomic_ref<int> a(i);
    ASSERT_EQ(a.load(), 0b0101);
    ASSERT_EQ(a.fetch_or(0b1100), 0b0101);
    ASSERT_EQ(a.load(), 0b1101);
    ASSERT_EQ(a.fetch_or(0b0000), 0b1101);
    ASSERT_EQ(a.load(), 0b1101);
}

TEST(core_device_atomic_test, atomic_fetch_xor) {
    int i = 0b0101;
    vecmem::device_atomic_ref<int> a(i);
    ASSERT_EQ(a.load(), 0b0101);
    ASSERT_EQ(a.fetch_xor(0b1100), 0b0101);
    ASSERT_EQ(a.load(), 0b1001);
    ASSERT_EQ(a.fetch_xor(0b0000), 0b1001);
    ASSERT_EQ(a.load(), 0b1001);
}
