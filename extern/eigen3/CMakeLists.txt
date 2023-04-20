# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
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
message( STATUS "Building Eigen3 as part of the TRACCC project" )

# Declare where to get Eigen3 from.
set( TRACCC_EIGEN_SOURCE
   "URL;https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2;URL_MD5;132dde48fe2b563211675626d29f1707"
   CACHE STRING "Source for Eigen, when built as part of this project" )
mark_as_advanced( TRACCC_EIGEN_SOURCE )
FetchContent_Declare( Eigen3 ${TRACCC_EIGEN_SOURCE} )

# Configure the Eigen build.
if( DEFINED CACHE{BUILD_TESTING} )
   set( _buildTestingValue ${BUILD_TESTING} )
endif()
set( BUILD_TESTING FALSE CACHE INTERNAL "Forceful setting of BUILD_TESTING" )
set( EIGEN_BUILD_DOC FALSE CACHE BOOL
   "Turn off the Eigen documentation build" )
set( EIGEN_TEST_NOQT TRUE CACHE BOOL
   "Don't set up Qt based Eigen tests/demos" )

# Get it into the current directory.
FetchContent_MakeAvailable( Eigen3 )

# Reset the BUILD_TESTING variable.
if( DEFINED _buildTestingValue )
   set( BUILD_TESTING ${_buildTestingValue} CACHE BOOL "Turn tests on/off"
      FORCE )
   unset( _buildTestingValue )
else()
   unset( BUILD_TESTING CACHE )
endif()

# Treat the Eigen headers as "system headers", to avoid getting warnings from
# them.
get_target_property( _incDirs eigen INTERFACE_INCLUDE_DIRECTORIES )
target_include_directories( eigen
   SYSTEM INTERFACE ${_incDirs} )
unset( _incDirs )

# For some wicked reason CUDA keeps complaining about the Eigen headers, even
# though they are set up from a "system include path". So I had to explicitly
# disable the warning triggered by those headers, for anything that uses them.
# This is pretty bad, as we're now blind to these types of warnings/errors. :-(
target_compile_options( eigen INTERFACE
   $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=20012> )
