# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.11 )
include( FetchContent )

# Silence various FetchContent warnings up to CMake 3.30.X.
if( POLICY CMP0135 )
   cmake_policy( SET CMP0135 NEW )
endif()
if( POLICY CMP0169 )
   cmake_policy( SET CMP0169 OLD )
endif()

# Tell the user what's happening.
message( STATUS "Building GoogleTest as part of the VecMem project" )

# Declare where to get GoogleTest from.
set( VECMEM_GOOGLETEST_SOURCE
   "URL;https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz;URL_MD5;7e11f6cfcf6498324ac82d567dcb891e"
   CACHE STRING "Source for GoogleTest, when built as part of VecMem" )
mark_as_advanced( VECMEM_GOOGLETEST_SOURCE )
FetchContent_Declare( GoogleTest ${VECMEM_GOOGLETEST_SOURCE} )

# Options used in the build of GoogleTest.
set( BUILD_GMOCK FALSE CACHE BOOL "Turn off the build of GMock" )
set( INSTALL_GTEST FALSE CACHE BOOL "Turn off the installation of GoogleTest" )
if( WIN32 )
   set( gtest_force_shared_crt TRUE CACHE BOOL
      "Use shared (DLL) run-time library, even with static libraries" )
endif()

# Silence some warnings with modern versions of CMake on macOS.
set( CMAKE_MACOSX_RPATH TRUE )

# Get it into the current directory.
FetchContent_Populate( GoogleTest )
set( CMAKE_FOLDER "vecmem/externals" )
add_subdirectory( "${googletest_SOURCE_DIR}" "${googletest_BINARY_DIR}"
   EXCLUDE_FROM_ALL )

# Set up aliases for the GTest targets with the same name that they have
# when we find GTest pre-installed.
add_library( GTest::gtest ALIAS gtest )
add_library( GTest::gtest_main ALIAS gtest_main )
