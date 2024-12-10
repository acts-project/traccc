# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Include the helper function(s).
include( algebra-plugins-functions )

# Basic flags for all build modes.
algebra_add_flag( CMAKE_SYCL_FLAGS "-Wall" )
algebra_add_flag( CMAKE_SYCL_FLAGS "-Wextra" )
algebra_add_flag( CMAKE_SYCL_FLAGS "-Wno-unknown-cuda-version" )
algebra_add_flag( CMAKE_SYCL_FLAGS "-Wshadow" )
algebra_add_flag( CMAKE_SYCL_FLAGS "-Wunused-local-typedefs" )
if( NOT WIN32 )
   algebra_add_flag( CMAKE_SYCL_FLAGS "-pedantic" )
endif()

# Avoid issues coming from MSVC<->DPC++ argument differences.
if( "${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC" )
   algebra_add_flag( CMAKE_SYCL_FLAGS
      "-Wno-unused-command-line-argument" )
endif()

# Fail on warnings, if asked for that behaviour.
if( ALGEBRA_PLUGINS_FAIL_ON_WARNINGS )
   algebra_add_flag( CMAKE_SYCL_FLAGS "-Werror" )
endif()
