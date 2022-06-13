# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.14 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building Detray as part of the TRACCC project" )

# Declare where to get Detray from.
set( TRACCC_DETRAY_SOURCE
   "URL;https://github.com/acts-project/detray/archive/refs/tags/v0.8.0.tar.gz;URL_MD5;17bc6260be77ffb8d69fd1fd06a4c22f"
   CACHE STRING "Source for Detray, when built as part of this project" )
mark_as_advanced( TRACCC_DETRAY_SOURCE )
FetchContent_Declare( Detray ${TRACCC_DETRAY_SOURCE} )

# Options used in the build of Detray.
set( DETRAY_CUSTOM_SCALARTYPE "float" CACHE STRING
   "Scalar type to use in the Detray code" )

set( DETRAY_BUILD_TESTING FALSE CACHE BOOL
   "Turn off the build of the Detray unit tests" )
set( DETRAY_EIGEN_PLUGIN TRUE CACHE BOOL
   "Turn on the build of the Detray Eigen code" )
set( DETRAY_VC_PLUGIN TRUE CACHE BOOL
   "Turn on the build of the Detray Vc code" )

set( DETRAY_SETUP_VECMEM FALSE CACHE BOOL
   "Do not set up VecMem as part of Detray" )
set( DETRAY_SETUP_ALGEBRA_PLUGINS FALSE CACHE BOOL
   "Do not set up Algebra Plugins as part of Detray" )
set( DETRAY_SETUP_DFELIBS FALSE CACHE BOOL
   "Do not set up dfelibs as part of Detray" )
set( DETRAY_SETUP_GOOGLETEST FALSE CACHE BOOL
   "Do not set up GoogleTest as part of Detray" )
set( DETRAY_SETUP_BENCHMARK FALSE CACHE BOOL
   "Do not set up Google Benchmark as part of Detray" )

# Get it into the current directory.
FetchContent_MakeAvailable( Detray )
