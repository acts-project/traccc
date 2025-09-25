# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2025 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.25 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building Acts as part of the TRACCC project" )

# Declare where to get Acts from.
# HACK: Go back to using a real Acts release when a version after
# v43.3.0 is released that incorporates Acts#4659
set( TRACCC_ACTS_SOURCE
   "GIT_REPOSITORY;https://github.com/acts-project/acts.git;GIT_TAG;dfa5b4fd97d1cf9674f3c580e5fc4cd7f0c4103f"
   CACHE STRING "Source for Acts, when built as part of this project" )
mark_as_advanced( TRACCC_ACTS_SOURCE )
FetchContent_Declare( Acts SYSTEM ${TRACCC_ACTS_SOURCE} )

# Options used in the build of Acts.
set( ACTS_SETUP_EIGEN3 FALSE CACHE BOOL
   "Do not set up Eigen in the Acts code, we do it in this project" )
set( ACTS_BUILD_PLUGIN_JSON TRUE CACHE BOOL
   "Build JSON plugin in Acts" )

# Get it into the current directory.
FetchContent_MakeAvailable( Acts )
