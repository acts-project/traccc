/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2020-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project inlcude(s)
#include "detray/definitions/detail/indexing.hpp"
#include "detray/grids/axis.hpp"
#include "detray/grids/serializer2.hpp"

// Detray test include(s)
#include "detray/test/utils/types.hpp"

// Vecmem include(s)
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s)
#include <gtest/gtest.h>

// System include(s)
#include <limits>

using namespace detray;

GTEST_TEST(detray_grid2, serialize_deserialize) {
    vecmem::host_memory_resource resource;

    axis2::regular<> r6{6u, -3.f, 7.f, resource};
    axis2::circular<> c12{12u, -3.f, 3.f, resource};

    serializer2 ser2;

    // Serializing
    dindex test = ser2.serialize(r6, c12, 0u, 0u);
    EXPECT_EQ(test, 0u);
    test = ser2.serialize(r6, c12, 5u, 0u);
    EXPECT_EQ(test, 5u);
    test = ser2.serialize(r6, c12, 0u, 1u);
    EXPECT_EQ(test, 6u);
    test = ser2.serialize(r6, c12, 5u, 2u);
    EXPECT_EQ(test, 17u);

    // Deserialize
    darray<dindex, 2> expected_array = {0u, 0u};
    darray<dindex, 2> test_array = ser2.deserialize(r6, c12, 0u);
    EXPECT_EQ(test_array, expected_array);
    expected_array = {5u, 0u};
    test_array = ser2.deserialize(r6, c12, 5u);
    EXPECT_EQ(test_array, expected_array);
    expected_array = {0u, 1u};
    test_array = ser2.deserialize(r6, c12, 6u);
    EXPECT_EQ(test_array, expected_array);
    expected_array = {5u, 2u};
    test_array = ser2.deserialize(r6, c12, 17u);
    EXPECT_EQ(test_array, expected_array);
}
