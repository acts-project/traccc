# SPDX-PackageName: "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: MPL-2.0

# Create the test executable from the individual test groups.
add_executable(test_cpu test_cpu_array_backend.cpp)

# Ensure that the tests are linked against the required libraries.
target_link_libraries(
    test_cpu
    PUBLIC
        covfie_core
        covfie_cpu
        GTest::gtest
        GTest::gtest_main
)
