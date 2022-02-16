# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Include the helper function(s).
include( traccc-functions )

# Basic flags for all build modes.
foreach( mode RELEASE RELWITHDEBINFO MINSIZEREL DEBUG )
   traccc_add_flag( CMAKE_SYCL_FLAGS_${mode} "-Wall" )
   traccc_add_flag( CMAKE_SYCL_FLAGS_${mode} "-Wextra" )
   traccc_add_flag( CMAKE_SYCL_FLAGS_${mode} "-Wno-unknown-cuda-version" )
   traccc_add_flag( CMAKE_SYCL_FLAGS_${mode} "-Wshadow" )
   traccc_add_flag( CMAKE_SYCL_FLAGS_${mode} "-Wunused-local-typedefs" )
endforeach()

# More rigorous tests for the Debug builds.
traccc_add_flag( CMAKE_SYCL_FLAGS_DEBUG "-Werror" )
if( NOT WIN32 )
   traccc_add_flag( CMAKE_SYCL_FLAGS_DEBUG "-pedantic" )
endif()

# Avoid issues coming from MSVC<->DPC++ argument differences.
if( "${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC" )
   foreach( mode RELEASE RELWITHDEBINFO MINSIZEREL DEBUG )
      traccc_add_flag( CMAKE_SYCL_FLAGS_${mode}
         "-Wno-unused-command-line-argument" )
   endforeach()
endif()
