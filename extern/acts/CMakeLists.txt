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
   "URL;https://github.com/krasznaa/acts/archive/refs/tags/v10.0.1.tar.gz;URL_MD5;fcd2f2f7b5810e34b8faa12634043be3"
   CACHE STRING "Source for Acts, when built as part of this project" )
mark_as_advanced( TRACCC_ACTS_SOURCE )
FetchContent_Declare( Acts ${TRACCC_ACTS_SOURCE} )

# Options used in the build of Acts.
set( ACTS_SETUP_EIGEN3 FALSE CACHE BOOL
   "Do not set up Eigen in the Acts code, we do it in this project" )
set( ACTS_BUILD_PLUGIN_JSON TRUE CACHE BOOL
   "Build JSON plugin in Acts" )

# Get it into the current directory.
FetchContent_MakeAvailable( Acts )
