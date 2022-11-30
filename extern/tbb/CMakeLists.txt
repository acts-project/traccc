# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022 CERN for the benefit of the ACTS project
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
message( STATUS "Building TBB as part of the TRACCC project" )

# Declare where to get TBB from.
set( TRACCC_TBB_SOURCE
   "URL;https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2021.7.0.tar.gz;URL_MD5;68e617448f71df02d8688c84d53778f6"
   CACHE STRING "Source for TBB, when built as part of this project" )
mark_as_advanced( TRACCC_TBB_SOURCE )
FetchContent_Declare( TBB ${TRACCC_TBB_SOURCE} )

# Options used in the build of TBB.
set( TBB_TEST FALSE CACHE BOOL "Turn off the TBB tests" )
set( TBB_STRICT FALSE CACHE BOOL "Do not throw errors on compiler warnings" )

# Get it into the current directory.
FetchContent_MakeAvailable( TBB )
