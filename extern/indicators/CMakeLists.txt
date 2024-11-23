# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2024 CERN for the benefit of the ACTS project
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
message( STATUS "Building indicators as part of the TRACCC project" )

# Declare where to get indicators from.
set( TRACCC_INDICATORS_SOURCE
   "URL;https://github.com/p-ranav/indicators/archive/refs/tags/v2.3.tar.gz;URL_MD5;317d2658fbb45d9d192f88ae0f1c9e9d"
   CACHE STRING "Source for indicators, when built as part of this project" )
mark_as_advanced( TRACCC_INDICATORS_SOURCE )
FetchContent_Declare( Indicators ${TRACCC_INDICATORS_SOURCE} )

# Configure the indicators build.
set( INDICATORS_BUILD_TESTS FALSE CACHE BOOL
   "Turn off the indicators tests build" )
set( INDICATORS_SAMPLES FALSE CACHE BOOL
   "Turn off the indicators samples build" )
set( INDICATORS_DEMO FALSE CACHE BOOL
   "Turn off the indicators demo build" )

# Get it into the current directory.
FetchContent_MakeAvailable( Indicators )

# Treat the indicators headers as "system headers", to avoid getting warnings
# from them.
get_target_property( _incDirs indicators INTERFACE_INCLUDE_DIRECTORIES )
target_include_directories( indicators
   SYSTEM INTERFACE ${_incDirs} )
unset( _incDirs )
