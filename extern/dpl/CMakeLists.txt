# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2024-2025 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.25 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building oneDPL as part of the TRACCC project" )

# Declare where to get DPL from.
set( TRACCC_DPL_SOURCE
   "URL;https://github.com/oneapi-src/oneDPL/archive/refs/tags/oneDPL-2022.7.1-release.tar.gz;URL_MD5;21d45dc27ba3133bfb282ec7383177f4"
   CACHE STRING "Source for DPL, when built as part of this project" )
mark_as_advanced( TRACCC_DPL_SOURCE )
FetchContent_Declare( DPL SYSTEM ${TRACCC_DPL_SOURCE} )

# Set the default oneDPL threading backend.
set( ONEDPL_BACKEND "dpcpp" CACHE STRING "oneDPL threading backend" )

# Get it into the current directory.
FetchContent_MakeAvailable( DPL )
