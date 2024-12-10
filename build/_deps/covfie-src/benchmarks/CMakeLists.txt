# SPDX-PackageName: "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: MPL-2.0

# All the tests here will require Google Test, so we will go ahead and find it.
find_package(benchmark CONFIG REQUIRED)

# All of these benchmarks rely on the Boost MP11 library.
find_package(Boost 1.71.0 REQUIRED)

# Set up the C++ compiler flags for the benchmarks.
include(covfie-compiler-options-cpp)

# Common benchmarking components must be build.
add_subdirectory(common)

# The CPU tests are only built if requested.
if(COVFIE_PLATFORM_CPU)
    add_subdirectory(cpu)
endif()

# The CUDA tests are only built if requested.
if(COVFIE_PLATFORM_CUDA)
    add_subdirectory(cuda)
endif()

# The OpenMP tests are only built if requested.
if(COVFIE_PLATFORM_OPENMP)
    add_subdirectory(openmp)
endif()
