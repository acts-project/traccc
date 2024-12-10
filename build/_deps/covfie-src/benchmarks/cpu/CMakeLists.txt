# SPDX-PackageName: "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: MPL-2.0

# Create the benchmark executable from the individual files.
add_executable(benchmark_cpu benchmark_cpu.cpp)

# Ensure that the tests are linked against the required libraries.
target_link_libraries(
    benchmark_cpu
    PUBLIC
        covfie_core
        covfie_cpu
        benchmark
        benchmark::benchmark
        Boost::headers
)
