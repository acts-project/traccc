# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.14 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building Thrust as part of the TRACCC project" )

# Declare where to get Thrust from.
set( TRACCC_THRUST_SOURCE
   "GIT_REPOSITORY;https://github.com/NVIDIA/thrust.git;GIT_TAG;1.15.0"
   CACHE STRING "Source for Thrust, when built as part of this project" )
mark_as_advanced( TRACCC_THRUST_SOURCE )
FetchContent_Declare( Thrust ${TRACCC_THRUST_SOURCE} )

# Options used in the build of Thrust.
set( THRUST_ENABLE_INSTALL_RULES TRUE CACHE BOOL
   "Install Thrust together with this project" )

# Get it into the current directory.
FetchContent_MakeAvailable( Thrust )
