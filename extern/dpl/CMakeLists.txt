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
message( STATUS "Building oneDPL as part of the TRACCC project" )

# Declare where to get DPL from.
set( TRACCC_DPL_SOURCE
   "URL;https://github.com/oneapi-src/oneDPL/archive/refs/tags/oneDPL-2022.6.0-rc1.tar.gz;URL_MD5;f52a2ed5c9e4cdb3c65c2465b50abecf"
   CACHE STRING "Source for DPL, when built as part of this project" )
mark_as_advanced( TRACCC_DPL_SOURCE )
FetchContent_Declare( DPL ${TRACCC_DPL_SOURCE} )

# Set the default oneDPL threading backend.
set( ONEDPL_BACKEND "dpcpp" CACHE STRING "oneDPL threading backend" )

# Get it into the current directory.
FetchContent_MakeAvailable( DPL )

# Treat the oneDPL headers as "system headers", to avoid getting warnings from
# them.
get_target_property( _incDirs oneDPL INTERFACE_INCLUDE_DIRECTORIES )
target_include_directories( oneDPL
   SYSTEM INTERFACE ${_incDirs} )
unset( _incDirs )