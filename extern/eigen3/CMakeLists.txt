# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.25 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building Eigen3 as part of the TRACCC project" )

# Declare where to get Eigen3 from.
set( TRACCC_EIGEN_SOURCE
   "URL;https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2;URL_MD5;132dde48fe2b563211675626d29f1707"
   CACHE STRING "Source for Eigen, when built as part of this project" )
mark_as_advanced( TRACCC_EIGEN_SOURCE )
FetchContent_Declare( Eigen3 SYSTEM ${TRACCC_EIGEN_SOURCE} )

# Configure the Eigen build.
set( BUILD_TESTING FALSE )
set( EIGEN_BUILD_DOC FALSE CACHE BOOL
   "Turn off the Eigen documentation build" )
set( EIGEN_TEST_NOQT TRUE CACHE BOOL
   "Don't set up Qt based Eigen tests/demos" )

# Get it into the current directory.
FetchContent_MakeAvailable( Eigen3 )
