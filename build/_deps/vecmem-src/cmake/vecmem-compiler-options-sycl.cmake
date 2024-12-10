# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Include the helper function(s).
include( vecmem-functions )

# Set up the used C++ standard(s).
set( CMAKE_SYCL_STANDARD 17 CACHE STRING "The (SYCL) C++ standard to use" )

# Basic flags for all build modes.
vecmem_add_flag( CMAKE_SYCL_FLAGS "-Wall" )
vecmem_add_flag( CMAKE_SYCL_FLAGS "-Wextra" )
vecmem_add_flag( CMAKE_SYCL_FLAGS "-Wno-unknown-cuda-version" )
vecmem_add_flag( CMAKE_SYCL_FLAGS "-Wshadow" )
vecmem_add_flag( CMAKE_SYCL_FLAGS "-Wunused-local-typedefs" )
if( NOT WIN32 )
   vecmem_add_flag( CMAKE_SYCL_FLAGS "-pedantic" )
endif()

# Avoid issues coming from MSVC<->DPC++ argument differences.
if( "${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC" )
   vecmem_add_flag( CMAKE_SYCL_FLAGS
      "-Wno-unused-command-line-argument" )
endif()

# Fail on warnings, if asked for that behaviour.
if( VECMEM_FAIL_ON_WARNINGS )
   vecmem_add_flag( CMAKE_SYCL_FLAGS "-Werror" )
endif()
