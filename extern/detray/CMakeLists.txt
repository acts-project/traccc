# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.25 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building Detray as part of the TRACCC project" )

# Declare where to get Detray from.
set( TRACCC_DETRAY_SOURCE
<<<<<<< Updated upstream
   "URL;https://github.com/acts-project/detray/archive/refs/tags/v0.103.0.tar.gz;URL_MD5;7df38072d676b63ee3f0f88bd84c0106"
=======
   "GIT_REPOSITORY;git@github.com:acts-project/detray.git;GIT_TAG;0a0cdc7f34e8ad048be147a40ea849d0a4a68b8a"
>>>>>>> Stashed changes
   CACHE STRING "Source for Detray, when built as part of this project" )
mark_as_advanced( TRACCC_DETRAY_SOURCE )
FetchContent_Declare( Detray SYSTEM ${TRACCC_DETRAY_SOURCE} )

# Options used in the build of Detray.
set( DETRAY_CUSTOM_SCALARTYPE "${TRACCC_CUSTOM_SCALARTYPE}" CACHE STRING
   "Scalar type to use in the Detray code" )

set( DETRAY_BUILD_UNITTESTS FALSE CACHE BOOL
   "Turn off the build of the Detray unit tests" )
set( DETRAY_BUILD_INTEGRATIONTESTS FALSE CACHE BOOL
   "Turn off the build of the Detray integration tests" )
set( DETRAY_BUILD_BENCHMARKS FALSE CACHE BOOL
   "Turn off the build of the Detray benchmarks" )
set( DETRAY_BUILD_CLI_TOOLS FALSE CACHE BOOL
   "Turn off the build of the Detray CLI tools" )
set( DETRAY_BUILD_TUTORIALS FALSE CACHE BOOL
   "Turn off the build of the Detray tutorials" )
set( DETRAY_EIGEN_PLUGIN TRUE CACHE BOOL
   "Turn on the build of the Detray Eigen code" )

set( DETRAY_BUILD_TEST_UTILS TRUE CACHE BOOL
   "Turn on the build of the Detray tests utilities, needed in this project" )
set( DETRAY_BUILD_VALIDATION_TOOLS FALSE CACHE BOOL
   "Turn off the build of the Detray validation tools" )

set( DETRAY_SETUP_VECMEM FALSE CACHE BOOL
   "Do not set up VecMem as part of Detray" )
set( DETRAY_SETUP_ALGEBRA_PLUGINS FALSE CACHE BOOL
   "Do not set up Algebra Plugins as part of Detray" )
set( DETRAY_SETUP_GOOGLETEST FALSE CACHE BOOL
   "Do not set up GoogleTest as part of Detray" )
set( DETRAY_SETUP_BENCHMARK FALSE CACHE BOOL
   "Do not set up Google Benchmark as part of Detray" )
set( DETRAY_SETUP_COVFIE FALSE CACHE BOOL
   "Do not set up covfie as part of Detray" )

# Get it into the current directory.
FetchContent_MakeAvailable( Detray )
