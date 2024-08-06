# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.14 )
include( FetchContent )

# Silence FetchContent warnings with CMake >=3.24.
if( POLICY CMP0135 )
    cmake_policy( SET CMP0135 NEW )
endif()

set (ENABLE_TESTS Off CACHE BOOL "")
set (ENABLE_EXAMPLES Off CACHE BOOL "")
set (ENABLE_REPRODUCERS Off CACHE BOOL "")
set (ENABLE_EXERCISES Off CACHE BOOL "")
set (ENABLE_DOCUMENTATION Off CACHE BOOL "")
set (ENABLE_BENCHMARKS Off CACHE BOOL "")

# Tell the user what's happening.
message( STATUS "Building Raja as part of the TRACCC project" )

# Declare where to get Raja from.
set( TRACCC_RAJA_SOURCE
        "GIT_REPOSITORY;https://github.com/LLNL/RAJA.git;GIT_TAG;v2023.06.1"
        CACHE STRING "Source for Raja, when built as part of this project" )
mark_as_advanced( TRACCC_RAJA_SOURCE )
FetchContent_Declare( Raja ${TRACCC_RAJA_SOURCE} )

# Options used in the build of Raja.
set( RAJA_ENABLE_INSTALL_RULES TRUE CACHE BOOL
        "Install Raja together with this project" )

#Turn off build testing
if( DEFINED CACHE{BUILD_TESTING} )
    set( _buildTestingValue ${BUILD_TESTING} )
endif()
set( BUILD_TESTING FALSE CACHE INTERNAL "Forceful setting of BUILD_TESTING" )


# Get it into the current directory.
FetchContent_MakeAvailable( Raja )

# Reset the BUILD_TESTING variable.
if( DEFINED _buildTestingValue )
    set( BUILD_TESTING ${_buildTestingValue} CACHE BOOL "Turn tests on/off"
            FORCE )
    unset( _buildTestingValue )
else()
    unset( BUILD_TESTING CACHE )
endif()