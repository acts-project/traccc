/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../common/memory_resource_name_gen.hpp"
#include "vecmem/containers/array.hpp"
#include "vecmem/memory/binary_page_memory_resource.hpp"
#include "vecmem/memory/contiguous_memory_resource.hpp"
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/memory/memory_resource.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <vector>

namespace {

/// Custom non-trivial type used in the tests.
struct TestType1 {
    TestType1(int a = 0, long b = 123) : m_a(a), m_b(b) {}
    int m_a;
    long m_b;
};
/// Helper operator for @c TestType1
bool operator==(const TestType1& value1, const TestType1& value2) {
    return ((value1.m_a == value2.m_a) && (value1.m_b == value2.m_b));
}

}  // namespace

/// Test case for @c vecmem::array
///
/// It provides a templated function doing the heavy lifting for the test(s).
///
class core_array_test
    : public testing::TestWithParam<vecmem::memory_resource*> {

protected:
    /// Function testing a particular array object.
    template <typename T, std::size_t N>
    void test_array(vecmem::array<T, N>& a) {

        // Fill the array with some simple values.
        for (std::size_t i = 0; i < a.size(); ++i) {
            a.at(i) = T(static_cast<int>(i));
        }

        // Check the contents using iterator based loops.
        {
            auto itr = a.begin();
            for (int i = 0; itr != a.end(); ++itr, ++i) {
                EXPECT_EQ(*itr, T(i));
            }
            auto ritr = a.rbegin();
            for (int i = static_cast<int>(a.size() - 1); ritr != a.rend();
                 ++ritr, --i) {
                EXPECT_EQ(*ritr, T(i));
            }
        }

        // Check its contents using a range based loop.
        {
            int i = 0;
            for (T value : a) {
                EXPECT_EQ(value, T(i++));
            }
        }

        // Fill the array with a constant value.
        static constexpr int VALUE = 123;
        a.fill(T(VALUE));
        for (T value : a) {
            EXPECT_EQ(value, T(VALUE));
        }

        // Make sure that it succeeded.
        if (!a.empty()) {
            EXPECT_EQ(a.front(), T(VALUE));
            EXPECT_EQ(a.back(), T(VALUE));
        }
        const std::vector<T> reference(a.size(), T(VALUE));
        EXPECT_EQ(a.size(), reference.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            EXPECT_EQ(a[i], reference[i]);
        }
    }

};  // class core_array_test

/// Test with a non-zero sized array whose size is fixed at compile time.
TEST_P(core_array_test, non_zero_compile_time) {

    vecmem::array<int, 10> a(*(GetParam()));
    test_array(a);
}

/// Test with a non-zero sized array whose size is specified at runtime
TEST_P(core_array_test, non_zero_runtime) {

    vecmem::array<TestType1> a(*(GetParam()), 20);
    test_array(a);
}

/// Test with a zero sized array whose size is fixed at compile time.
TEST_P(core_array_test, zero_compile_time) {

    vecmem::array<unsigned int, 0> a(*(GetParam()));
    test_array(a);
}

/// Test with a zero sized array whose size is specified at runtime
TEST_P(core_array_test, zero_runtime) {

    vecmem::array<TestType1> a(*(GetParam()), 0);
    test_array(a);
}

namespace {

/// Simple test type
struct TestType2 {
    /// Constructor
    TestType2(int a = 10) : m_value(a + 1), m_pointer(nullptr) {}
    /// Destructor
    ~TestType2() {
        if (m_pointer != nullptr) {
            *m_pointer -= 1;
        }
    }
    /// Simple member variable
    int m_value;
    /// Pointer member variable
    int* m_pointer;
};  // struct TestType2

}  // namespace

/// Make sure that a non-trivial constructor and destructor is executed properly
TEST_P(core_array_test, constructor_destructor) {

    int dummy = 5;
    {
        vecmem::array<TestType2, 10> a(*(GetParam()));
        EXPECT_EQ(a.front().m_value, 11);
        a.back().m_pointer = &dummy;
    }
    EXPECT_EQ(dummy, 4);
}

// Memory resources to use in the test.
static vecmem::host_memory_resource host_resource;
static vecmem::binary_page_memory_resource binary_resource(host_resource);
static vecmem::contiguous_memory_resource contiguous_resource(host_resource,
                                                              20000);

// Instantiate the test suite.
INSTANTIATE_TEST_SUITE_P(core_array_tests, core_array_test,
                         testing::Values(&host_resource, &binary_resource,
                                         &contiguous_resource),
                         vecmem::testing::memory_resource_name_gen(
                             {{&host_resource, "host_resource"},
                              {&binary_resource, "binary_resource"},
                              {&contiguous_resource, "contiguous_resource"}}));
