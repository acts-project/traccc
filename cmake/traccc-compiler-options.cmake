# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Guard against multiple includes.
include_guard( GLOBAL )

# Include the helper function(s).
include( traccc-functions )

# Set the language standards to use.
set( CMAKE_CXX_STANDARD 17 CACHE STRING "The (Host) C++ standard to use" )
set( CMAKE_CUDA_STANDARD 17 CACHE STRING "The (CUDA) C++ standard to use" )

# Turn on a number of warnings for the "known compilers".
if( ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" ) OR
    ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" ) )

   # Basic flags for all major build modes.
   traccc_add_flag( CMAKE_CXX_FLAGS "-Wall" )
   traccc_add_flag( CMAKE_CXX_FLAGS "-Wextra" )

endif()

# Set the CUDA architecture to build code for.
set( CMAKE_CUDA_ARCHITECTURES "52" CACHE STRING
   "CUDA architectures to build device code for" )

# Make CUDA generate debug symbols for the device code as well in a debug
# build.
traccc_add_flag( CMAKE_CUDA_FLAGS_DEBUG "-G" )
