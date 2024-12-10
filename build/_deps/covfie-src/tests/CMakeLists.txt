# SPDX-PackageName: "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: MPL-2.0

# All the tests here will require Google Test, so we will go ahead and set it
# up.
find_package(GTest CONFIG REQUIRED)

# Set up the C++ compiler flags for the tests.
include(covfie-compiler-options-cpp)

# The testing utils are always built.
add_subdirectory(utils)

# The core library tests are always built (if tests are built at all).
add_subdirectory(core)

# The CPU tests are only built if requested.
if(COVFIE_PLATFORM_CPU)
    add_subdirectory(cpu)
endif()

# The CUDA tests are only built if requested.
if(COVFIE_PLATFORM_CUDA)
    add_subdirectory(cuda)
endif()
