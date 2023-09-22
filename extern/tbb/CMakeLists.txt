# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022-2023 CERN for the benefit of the ACTS project
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
message( STATUS "Building TBB as part of the TRACCC project" )

# Declare where to get TBB from.
set( TRACCC_TBB_SOURCE
   "URL;https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2021.7.0.tar.gz;URL_MD5;68e617448f71df02d8688c84d53778f6"
   CACHE STRING "Source for TBB, when built as part of this project" )
mark_as_advanced( TRACCC_TBB_SOURCE )
FetchContent_Declare( TBB ${TRACCC_TBB_SOURCE} )

# Options used in the build of TBB.
set( TBB_TEST FALSE CACHE BOOL "Turn off the TBB tests" )
set( TBB_STRICT FALSE CACHE BOOL "Do not throw errors on compiler warnings" )

# Get it into the current directory. Setting TBB_DIR locally to an empty string
# is necessary so that incremental rebuilds would not get confused about the
# TBB_DIR cache variable set a little further down in this file. (The TBB CMake
# configuration is plenty weird...)
set( TBB_DIR "" )
FetchContent_MakeAvailable( TBB )

# Make CMake pick up a no-op TBBConfig.cmake file in any possible
# find_package(TBB) call. (Mostly needed in case other externals also need
# TBB.)
include( CMakePackageConfigHelpers )
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/TBBConfig.cmake.in"
   "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/TBBConfig.cmake"
   @ONLY )
write_basic_package_version_file(
   "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/TBBConfig-version.cmake"
   VERSION "2021.7.0"
   COMPATIBILITY "AnyNewerVersion" )
set( TBB_DIR "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}" CACHE PATH
   "TBBConfig.cmake location" )
