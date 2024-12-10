# SPDX-PackageName: "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: MPL-2.0

# Load the OpenMP platform.
find_package(OpenMP COMPONENTS CXX REQUIRED)

# Create the benchmark executable from the individual files.
add_executable(benchmark_openmp benchmark_openmp.cpp)

# Ensure that the tests are linked against the required libraries.
target_link_libraries(
    benchmark_openmp
    PUBLIC
        covfie_core
        covfie_cpu
        benchmark
        OpenMP::OpenMP_CXX
        benchmark::benchmark
        Boost::headers
)
