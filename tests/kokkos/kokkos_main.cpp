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

/// Main function for the Kokkos unit test(s)
///
/// In order to ensure that Kokkos is initialised and finalised just once in the
/// test jobs, we need to use a custom @c main function instead of the GoogeTest
/// provided one.
///
/// @param argc The number of command line arguments
/// @param argv The array of command line arguments
/// @return @c 0 if successful, something else if not
///
int main(int argc, char** argv) {

    // Initialise both Kokkos and GoogleTest.
    Kokkos::initialize(argc, argv);
    testing::InitGoogleTest(&argc, argv);

    // Run all of the tests.
    const int r = RUN_ALL_TESTS();

    // Finalise Kokkos.
    Kokkos::finalize();

    // Return the appropriate code.
    return r;
}
