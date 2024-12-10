# SPDX-PackageName: "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
#
# SPDX-License-Identifier: MPL-2.0

# FindCUDAToolkit needs at least CMake 3.17.
cmake_minimum_required(VERSION 3.17)

# Include the helper function(s).
include(covfie-functions)

# Figure out the properties of CUDA being used.
find_package(CUDAToolkit REQUIRED)

# Set the architecture to build code for.
set(CMAKE_CUDA_ARCHITECTURES
    "52"
    CACHE STRING
    "CUDA architectures to build device code for"
)

# Turn on the correct setting for the __cplusplus macro with MSVC.
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
    covfie_add_flag( CMAKE_CUDA_FLAGS "-Xcompiler /Zc:__cplusplus" )
endif()

if("${CMAKE_CUDA_COMPILER_ID}" MATCHES "NVIDIA")
    # Make CUDA generate debug symbols for the device code as well in a debug
    # build.
    covfie_add_flag( CMAKE_CUDA_FLAGS_DEBUG "-G" )
endif()

covfie_add_flag( CMAKE_CUDA_FLAGS "-Wfloat-conversion" )
covfie_add_flag( CMAKE_CUDA_FLAGS "-Wconversion" )

# Fail on warnings, if asked for that behaviour.
if(COVFIE_FAIL_ON_WARNINGS)
    if(
        (
            "${CUDAToolkit_VERSION}"
                VERSION_GREATER_EQUAL
                "10.2"
        )
        AND (
            "${CMAKE_CUDA_COMPILER_ID}"
                MATCHES
                "NVIDIA"
        )
    )
        covfie_add_flag( CMAKE_CUDA_FLAGS "-Werror all-warnings" )
    elseif("${CMAKE_CUDA_COMPILER_ID}" MATCHES "Clang")
        covfie_add_flag( CMAKE_CUDA_FLAGS "-Werror" )
    endif()
endif()
