/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/contiguous_memory_resource.hpp"
#include "vecmem/memory/host_memory_resource.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <cstddef>

/// Test case for @c vecmem::contiguous_memory_resource
class core_contiguous_memory_resource_test : public testing::Test {

protected:
    /// The base memory resource
    vecmem::host_memory_resource m_upstream;
    /// The contiguous memory resource
    vecmem::contiguous_memory_resource m_resource{m_upstream, 1048576};

};  // class core_contiguous_memory_resource_test

/// Test the dumb allocation of a bunch of vectors
TEST_F(core_contiguous_memory_resource_test, allocations) {

// Skip this test with MSVC, in debug builds. As MSVC pads vectors in a way
// that makes this test fail.
#if defined(_MSC_VER) && defined(_DEBUG)
    GTEST_SKIP();
#else

    /// Fixed size for the test vectors
    static constexpr std::size_t VECTOR_SIZE = 100;

    // Create vectors, and check that they are put into memory as we would
    // expect it.
    vecmem::vector<int> vec1(VECTOR_SIZE, &m_resource);

    vecmem::vector<char> vec2(VECTOR_SIZE, &m_resource);
    EXPECT_GE(static_cast<void*>(&*(vec2.begin())),
              static_cast<void*>(&*(vec1.end())));

    vecmem::vector<double> vec3(VECTOR_SIZE, &m_resource);
    EXPECT_GE(static_cast<void*>(&*(vec3.begin())),
              static_cast<void*>(&*(vec2.end())));

    vecmem::vector<float> vec4(VECTOR_SIZE, &m_resource);
    EXPECT_GE(static_cast<void*>(&*(vec4.begin())),
              static_cast<void*>(&*(vec3.end())));

    vecmem::vector<int> vec5(VECTOR_SIZE, &m_resource);
    EXPECT_GE(static_cast<void*>(&*(vec5.begin())),
              static_cast<void*>(&*(vec4.end())));

#endif  // MSVC debug build...
}
