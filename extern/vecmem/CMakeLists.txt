# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2025 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.25 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building VecMem as part of the TRACCC project" )

# Declare where to get VecMem from.
set( TRACCC_VECMEM_SOURCE
   "URL;https://github.com/acts-project/vecmem/archive/refs/tags/v1.18.0.tar.gz;URL_MD5;288ec7bb30e209ab5cc2e7f4a209c7ac"
   CACHE STRING "Source for VecMem, when built as part of this project" )
mark_as_advanced( TRACCC_VECMEM_SOURCE )
FetchContent_Declare( VecMem SYSTEM ${TRACCC_VECMEM_SOURCE} )

# Options used in the build of VecMem.
set( VECMEM_BUILD_TESTING FALSE CACHE BOOL
   "Turn off the build of the VecMem unit tests" )

# Get it into the current directory.
FetchContent_MakeAvailable( VecMem )
