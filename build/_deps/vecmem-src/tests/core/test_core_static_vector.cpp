/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/containers/static_vector.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

/// Test case for @c vecmem::static_vector
template <typename T>
class core_static_vector_test : public testing::Test {};

namespace {

/// "Complex" type used in the bulk of the tests
struct TestType1 {
    TestType1(int a = 0, long b = 123l) : m_a(a), m_b(b) {}
    int m_a;
    long m_b;
};  // struct TestType1

/// Helper operator for comparing two @c TestType1 objects
bool operator==(const TestType1& v1, const TestType1& v2) {
    return ((v1.m_a == v2.m_a) && (v1.m_b == v2.m_b));
}

/// Helper struct/function for printing nice names for the tested types
struct tested_names {
    template <typename T>
    static std::string GetName(int i) {
        switch (i) {
            case 0:
                return "int";
                break;
            case 1:
                return "long";
                break;
            case 2:
                return "float";
                break;
            case 3:
                return "double";
                break;
            case 4:
                return "TestType1";
                break;
            default:
                return "Unknown";
        }
    }
};

}  // namespace

/// Test suite for primitive types.
typedef testing::Types<int, long, float, double, TestType1> tested_types;
TYPED_TEST_SUITE(core_static_vector_test, tested_types, tested_names);

namespace {
/// Helper function for comparing the value of non-integral primitive types
template <typename T>
typename std::enable_if<!std::is_integral<T>::value, bool>::type almost_equal(
    const T& v1, const T& v2) {
    return (std::abs(v1 - v2) < 0.001);
}
/// Helper function for comparing the value of integral primitive types
template <typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type almost_equal(
    const T& v1, const T& v2) {
    return (v1 == v2);
}
/// Specialisation for comparing @c TestType1 objects
template <>
bool almost_equal<TestType1>(const TestType1& v1, const TestType1& v2) {
    return (v1 == v2);
}
}  // namespace

/// Helper macro for comparing two vectors.
#define EXPECT_EQ_VEC(v1, v2)                                            \
    EXPECT_EQ(v1.size(), v2.size());                                     \
    EXPECT_TRUE(std::equal(std::begin(v1), std::end(v1), std::begin(v2), \
                           ::almost_equal<TypeParam>))

/// Test the constructor with a size
TYPED_TEST(core_static_vector_test, constructor_with_size) {

    // Create a vector.
    vecmem::static_vector<TypeParam, 100> v(10);
    EXPECT_EQ(v.size(), 10);

    // Make sure that it's elements were created as expected.
    for (const TypeParam& value : v) {
        EXPECT_EQ(value, TypeParam());
    }
}

/// Test the constructor with a size and a custom value
TYPED_TEST(core_static_vector_test, constructor_with_size_and_value) {

    // Create a vector.
    static const TypeParam DEFAULT_VALUE = 10;
    vecmem::static_vector<TypeParam, 100> v(10, DEFAULT_VALUE);
    EXPECT_EQ(v.size(), 10);

    // Make sure that it's elements were created as expected.
    for (const TypeParam& value : v) {
        EXPECT_EQ(value, DEFAULT_VALUE);
    }
}

/// Test the constructor with a range of values
TYPED_TEST(core_static_vector_test, constructor_with_iterators) {

    // Create a reference vector.
    const std::vector<TypeParam> ref = {1, 23, 64, 66, 23, 64, 99};

    // Create the test vector based on it.
    const vecmem::static_vector<TypeParam, 100> test(ref.begin(), ref.end());
    EXPECT_EQ_VEC(ref, test);
}

/// Test the default constructor
TYPED_TEST(core_static_vector_test, default_constructor) {

    // Create a default vector.
    const vecmem::static_vector<TypeParam, 100> test1;
    EXPECT_EQ(test1.size(), 0);

    // Create a default vector with zero capacity.
    const vecmem::static_vector<TypeParam, 0> test2;
    EXPECT_EQ(test2.size(), 0);
}

/// Test the copy constructor
TYPED_TEST(core_static_vector_test, copy_constructor) {

    // Create a reference vector.
    static const TypeParam DEFAULT_VALUE = 123;
    vecmem::static_vector<TypeParam, 100> ref(10, DEFAULT_VALUE);

    // Create a copy.
    vecmem::static_vector<TypeParam, 100> copy(ref);

    // Check the copy.
    EXPECT_EQ_VEC(ref, copy);
}

/// Test the element access functions and operators
TYPED_TEST(core_static_vector_test, element_access) {

    // Create a vector.
    vecmem::static_vector<TypeParam, 100> v(10);

    // Modify its elements.
    for (std::size_t i = 0; i < v.size(); ++i) {
        v.at(i) = TypeParam(static_cast<int>(i));
    }

    // Check that the settings "took".
    for (std::size_t i = 0; i < v.size(); ++i) {
        EXPECT_EQ(v[i], TypeParam(static_cast<int>(i)));
    }

    // Test the front() and back() functions.
    EXPECT_EQ(v.front(), TypeParam(0));
    EXPECT_EQ(v.back(), TypeParam(9));

    // Make sure that the vector points to a meaningful place.
    EXPECT_EQ(&(v.front()), v.data());
}

/// Test modifying an existing vector
TYPED_TEST(core_static_vector_test, modifications) {

    // Here we perform the same operations on a reference and on a test vector.
    // Assuming that std::vector would always behave correctly, so we just need
    // to test vecmem::static_vector against it.

    // Fill the vectors with some simple content.
    std::vector<TypeParam> ref(50);
    vecmem::static_vector<TypeParam, 100> test(50);
    EXPECT_EQ_VEC(ref, test);
    for (std::size_t i = 0; i < ref.size(); ++i) {
        ref[i] = TypeParam(static_cast<int>(i));
        test[i] = TypeParam(static_cast<int>(i));
    }
    EXPECT_EQ_VEC(ref, test);

    // Add a single element to the end of them.
    ref.push_back(TypeParam(60));
    test.push_back(TypeParam(60));
    EXPECT_EQ_VEC(ref, test);

    // Do the same, just in a slightly different colour.
    ref.emplace_back(TypeParam(70));
    test.emplace_back(TypeParam(70));
    EXPECT_EQ_VEC(ref, test);

    // Insert a single element in the middle of them.
    ref.insert(ref.begin() + 20, TypeParam(15));
    test.insert(test.begin() + 20, TypeParam(15));
    EXPECT_EQ_VEC(ref, test);

    // Emplace a single element in the middle of them.
    ref.emplace(ref.begin() + 15, TypeParam(55));
    test.emplace(test.begin() + 15, TypeParam(55));
    EXPECT_EQ_VEC(ref, test);

    // Remove one element from them.
    ref.erase(ref.begin() + 30);
    test.erase(test.begin() + 30);
    EXPECT_EQ_VEC(ref, test);

    // Remove a range of elements from them.
    ref.erase(ref.begin() + 10, ref.begin() + 25);
    test.erase(test.begin() + 10, test.begin() + 25);
    EXPECT_EQ_VEC(ref, test);

    // Insert N copies of the same value in them.
    ref.insert(ref.begin() + 13, 10, TypeParam(34));
    test.insert(test.begin() + 13, 10, TypeParam(34));
    EXPECT_EQ_VEC(ref, test);

    // Insert a range of new values.
    static const std::vector<TypeParam> ins = {33, 44, 55};
    ref.insert(ref.begin() + 24, ins.begin(), ins.end());
    test.insert(test.begin() + 24, ins.begin(), ins.end());
    EXPECT_EQ_VEC(ref, test);

    // Reduce the size of them.
    ref.resize(5);
    test.resize(5);
    EXPECT_EQ_VEC(ref, test);

    // Expand them.
    ref.resize(20);
    test.resize(20);
    EXPECT_EQ_VEC(ref, test);

    // Expand them using a specific fill value.
    ref.resize(30, TypeParam(93));
    test.resize(30, TypeParam(93));
    EXPECT_EQ_VEC(ref, test);
}

/// Test the capacity functions of @c vecmem::static_vector
TYPED_TEST(core_static_vector_test, capacity) {

    // Create a simple vector.
    vecmem::static_vector<TypeParam, 100> v;

    // Simple checks on the empty vector.
    EXPECT_EQ(v.empty(), true);
    EXPECT_EQ(v.size(), 0);
    EXPECT_EQ(v.max_size(), 100);
    EXPECT_EQ(v.capacity(), 100);

    // Resize it, and test it again.
    v.resize(50);
    EXPECT_EQ(v.empty(), false);
    EXPECT_EQ(v.size(), 50);
    EXPECT_EQ(v.max_size(), 100);
    EXPECT_EQ(v.capacity(), 100);

    // Make sure that the (no-op) reserve function can be called.
    v.reserve(70);
}

namespace {

/// Complex type used in one of the tests.
struct TestType2 {
    TestType2(int a) : m_a(a + 1), m_b(nullptr) {}
    ~TestType2() {
        if (m_b != nullptr) {
            --(*m_b);
        }
    }
    int m_a;
    int* m_b;
};  // struct TestType2

}  // namespace

/// Simple test fixture for the non-typed-tests
class core_static_vector_test_complex : public testing::Test {};

/// Test a value type with a non-trivial constructor and destructor
TEST_F(core_static_vector_test_complex, non_trivial_value) {

    // Create the test vector.
    vecmem::static_vector<TestType2, 100> v;

    // Fill it with some custom values.
    v.insert(v.begin(), 20, TestType2(10));
    EXPECT_EQ(v.size(), 20);
    for (const TestType2& value : v) {
        EXPECT_EQ(value.m_a, 11);
        EXPECT_EQ(value.m_b, nullptr);
    }

    // Make sure that the destructor is called on the vector elements.
    int dummy = 10;
    v[0].m_b = &dummy;
    v[5].m_b = &dummy;
    v.clear();
    EXPECT_EQ(dummy, 8);
}
