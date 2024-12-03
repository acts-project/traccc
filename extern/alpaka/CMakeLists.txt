# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.25 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building Alpaka as part of the TRACCC project" )

# Declare where to get Alpaka from.
set( TRACCC_ALPAKA_SOURCE
   "URL;https://github.com/alpaka-group/alpaka/archive/refs/tags/1.2.0.tar.gz;URL_MD5;0e2dcf4ddf7987532835d7b2b3680843"
   CACHE STRING "Source for Alpaka, when built as part of this project" )
mark_as_advanced( TRACCC_ALPAKA_SOURCE )
FetchContent_Declare( Alpaka SYSTEM ${TRACCC_ALPAKA_SOURCE} )

# Turn off build testing.
set( BUILD_TESTING FALSE )

# Get it into the current directory.
FetchContent_MakeAvailable( Alpaka )
