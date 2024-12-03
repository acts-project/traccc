# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.25 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building rocThrust as part of the TRACCC project" )

# Declare where to get rocThrust from.
set( TRACCC_ROCTHRUST_SOURCE
   "URL;https://github.com/ROCm/rocThrust/archive/refs/tags/rocm-6.1.1.tar.gz;URL_MD5;038abf313688c00555fe1efc51e1307b"
   CACHE STRING "Source for rocThrust, when built as part of this project" )
set( TRACCC_ROCTHRUST_PATCH
   "PATCH_COMMAND;patch;-p1;<;${CMAKE_CURRENT_SOURCE_DIR}/rocm-6.1.1.patch"
   CACHE STRING "Patch for rocThrust, when built as part of this project" )
mark_as_advanced( TRACCC_ROCTHRUST_SOURCE TRACCC_ROCTHRUST_PATCH )
FetchContent_Declare( rocThrust SYSTEM
   ${TRACCC_ROCTHRUST_SOURCE}
   ${TRACCC_ROCTHRUST_PATCH} )

# Settings for the rocThrust build.
set( BUILD_TESTING FALSE )
set( HIP_COMPILER "clang" )
set( HIP_CXX_COMPILER "hipcc" )

# Get it into the current directory.
FetchContent_MakeAvailable( rocThrust )
