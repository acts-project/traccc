/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/host_memory_resource.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

/// Test case for @c vecmem::vector
class core_vector_test : public testing::Test {

protected:
    /// Memory resource to use in the tests
    vecmem::host_memory_resource m_resource;

};  // class core_vector_test

namespace {

/// Simple test type
struct TestType1 {
    /// Constructor
    TestType1(int a, int& b) : m_value(a + 1), m_pointer(&b) {}
    /// Destructor
    ~TestType1() { *m_pointer -= 1; }
    /// Simple member variable
    int m_value;
    /// Pointer member variable
    int* m_pointer;
};  // struct TestType1

}  // namespace

/// Make sure that a non-trivial constructor and destructor is executed properly
TEST_F(core_vector_test, constructor_destructor) {

    int dummy = 5;
    {
        vecmem::vector< ::TestType1> test_vector(&m_resource);
        test_vector.emplace_back(10, dummy);
        EXPECT_EQ(test_vector.back().m_value, 11);
    }
    EXPECT_EQ(dummy, 4);
}
