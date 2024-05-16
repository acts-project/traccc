# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.14 )
include( FetchContent )

# Silence FetchContent warnings with CMake >=3.24.
if( POLICY CMP0135 )
   cmake_policy( SET CMP0135 NEW )
endif()

# Tell the user what's happening.
message( STATUS "Building rocThrust as part of the TRACCC project" )

# Declare where to get rocThrust from.
set( TRACCC_ROCTHRUST_SOURCE
   "GIT_REPOSITORY;https://github.com/ROCm/rocThrust.git;GIT_TAG;rocm-6.1.0"
   CACHE STRING "Source for Thrust, when built as part of this project" )
mark_as_advanced( TRACCC_ROCTHRUST_SOURCE )
FetchContent_Declare( rocThrust ${TRACCC_ROCTHRUST_SOURCE} )

#Turn off build testing
if( DEFINED CACHE{BUILD_TESTING} )
   set( _buildTestingValue ${BUILD_TESTING} )
endif()
set( BUILD_TESTING FALSE CACHE INTERNAL "Forceful setting of BUILD_TESTING" )

# Get it into the current directory.
FetchContent_MakeAvailable( rocThrust )

# Reset the BUILD_TESTING variable.
if( DEFINED _buildTestingValue )
   set( BUILD_TESTING ${_buildTestingValue} CACHE BOOL "Turn tests on/off"
      FORCE )
   unset( _buildTestingValue )
else()
   unset( BUILD_TESTING CACHE )
endif()
