# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.11 )
include( FetchContent )

# Silence FetchContent warnings with CMake >=3.24.
if( POLICY CMP0135 )
   cmake_policy( SET CMP0135 NEW )
endif()

# Tell the user what's happening.
message( STATUS "Building Acts as part of the TRACCC project" )

# Declare where to get Acts from.
set( TRACCC_ACTS_SOURCE
   "URL;https://github.com/acts-project/acts/archive/refs/tags/v37.3.0.tar.gz;URL_MD5;95cf54fe7986cece23ca7004d1138b01"
   CACHE STRING "Source for Acts, when built as part of this project" )
mark_as_advanced( TRACCC_ACTS_SOURCE )

# Mark the import as a system library on modern CMake versions
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.25.0)
   set(TRACCC_ACTS_SOURCE_FULL "${TRACCC_ACTS_SOURCE};SYSTEM")
else()
   set(TRACCC_ACTS_SOURCE_FULL "${TRACCC_ACTS_SOURCE}")
endif()
mark_as_advanced( TRACCC_ACTS_SOURCE_FULL )

FetchContent_Declare( Acts ${TRACCC_ACTS_SOURCE_FULL} )

# Options used in the build of Acts.
set( ACTS_SETUP_EIGEN3 FALSE CACHE BOOL
   "Do not set up Eigen in the Acts code, we do it in this project" )
set( ACTS_BUILD_PLUGIN_JSON TRUE CACHE BOOL
   "Build JSON plugin in Acts" )

# Get it into the current directory.
FetchContent_MakeAvailable( Acts )
