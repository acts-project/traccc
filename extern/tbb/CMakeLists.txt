# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.24 )
include( FetchContent )

# Silence FetchContent warnings with CMake >=3.24.
if( POLICY CMP0135 )
   cmake_policy( SET CMP0135 NEW )
endif()

# Tell the user what's happening.
message( STATUS "Building TBB as part of the TRACCC project" )

# Declare where to get TBB from.
set( TRACCC_TBB_SOURCE
   "URL;https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2021.13.0.tar.gz;URL_MD5;f287cd007240a838286ff13e7deaee12"
   CACHE STRING "Source for TBB, when built as part of this project" )
mark_as_advanced( TRACCC_TBB_SOURCE )
FetchContent_Declare( TBB ${TRACCC_TBB_SOURCE} OVERRIDE_FIND_PACKAGE )

# Options used in the build of TBB.
set( TBB_TEST FALSE CACHE BOOL "Turn off the TBB tests" )
set( TBB_STRICT FALSE CACHE BOOL "Do not throw errors on compiler warnings" )

# Get it into the current directory.
FetchContent_MakeAvailable( TBB )

# Treat the TBB headers as "system headers", to avoid getting warnings from
# them.
get_target_property( _incDirs tbb INTERFACE_INCLUDE_DIRECTORIES )
target_include_directories( tbb
   SYSTEM INTERFACE ${_incDirs} )
unset( _incDirs )
