/**
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Kokkos include(s).
#include <Kokkos_Core.hpp>

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <vector>

/// Trivial test for a host-run parallel-for.
GTEST_TEST(KokkosBasic, ParalleFor) {

    // Allocate an integer vector.
    std::vector<int> test_vec(100);
    int* test_vec_ptr = test_vec.data();

    // Set all elements to some value. Forecfully running serially on the host.
    // In case the default execution space would be CUDA. (Which would not be
    // compatible with the host-allocated std::vector of course.)
    Kokkos::parallel_for(
        "test_vec_init",
        Kokkos::RangePolicy<Kokkos::Serial>(0, test_vec.size()),
        KOKKOS_LAMBDA(int i) { test_vec_ptr[i] = 1; });

    // Check that we succeeded.
    for (int v : test_vec) {
        EXPECT_EQ(v, 1);
    }
}
