# SPDX-PackageName: "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: MPL-2.0

# We use the Boost filesystem for the creation of some temporary files, so
# we'll load that as well.
find_package(Boost 1.71.0 REQUIRED COMPONENTS filesystem)

# Create the test executable from the individual test groups.
add_executable(
    test_core
    test_utility.cpp
    test_constant_field.cpp
    test_transformer_affine.cpp
    test_transformer_clamp.cpp
    test_binary_io.cpp
    test_identity_backend.cpp
    test_nearest_neighbour_interpolator.cpp
    test_linear_interpolator.cpp
    test_algebra.cpp
    test_static_permutation.cpp
    test_utility_numeric.cpp
    test_array_binary_io.cpp
    test_build.cpp
    test_canonical.cpp
    test_atlas_like_io.cpp
    test_covariant_cast.cpp
)

# Ensure that the tests are linked against the required libraries.
target_link_libraries(
    test_core
    PUBLIC
        covfie_core
        GTest::gtest
        GTest::gtest_main
        Boost::filesystem
        testing_utils
)
