# SPDX-PackageName: "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: MPL-2.0

# Set up the C++ compiler flags for the examples.
include(covfie-compiler-options-cpp)

# The common tools should be available for examples on all platforms.
add_subdirectory(common)

# Enable position-independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The Boost log and options libraries are pretty universal, so we'll load them
# here.
find_package(
    Boost
    1.71.0
    REQUIRED
    COMPONENTS
        log
        log_setup
        program_options
)

# As always, the core examples are built any time the examples are.
add_subdirectory(core)

# The CPU examples are built only if that is a requested platform.
if(COVFIE_PLATFORM_CPU)
    add_subdirectory(cpu)
endif()

# And of course the same goes for the CUDA examples.
if(COVFIE_PLATFORM_CUDA)
    add_subdirectory(cuda)
endif()
