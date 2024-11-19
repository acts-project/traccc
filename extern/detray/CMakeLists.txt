# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
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
message( STATUS "Building Detray as part of the TRACCC project" )

# Declare where to get Detray from.
set( TRACCC_DETRAY_SOURCE
"URL;https://github.com/acts-project/detray/archive/refs/tags/v0.82.0.tar.gz;URL_MD5;09d777ce126c547f5c3d4ec509f9a0c0"
   CACHE STRING "Source for Detray, when built as part of this project" )

mark_as_advanced( TRACCC_DETRAY_SOURCE )

# Mark the import as a system library on modern CMake versions
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.25.0)
   set(TRACCC_DETRAY_SOURCE_FULL "${TRACCC_DETRAY_SOURCE};SYSTEM")
else()
   set(TRACCC_DETRAY_SOURCE_FULL "${TRACCC_DETRAY_SOURCE}")
endif()
mark_as_advanced( TRACCC_DETRAY_SOURCE_FULL )

FetchContent_Declare( Detray ${TRACCC_DETRAY_SOURCE_FULL} )

# Options used in the build of Detray.
set( DETRAY_CUSTOM_SCALARTYPE "float" CACHE STRING
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

# Needed for 'performance', 'simulation', 'examples', 'tests' and 'benchmarks'
if( TRACCC_BUILD_IO OR TRACCC_BUILD_EXAMPLES OR (BUILD_TESTING AND TRACCC_BUILD_TESTING) OR TRACCC_BUILD_BENCHMARKS )
   set( DETRAY_BUILD_TEST_UTILS TRUE CACHE BOOL
      "Turn on the build of the Detray tests utilities if needed" )
else()
   set( DETRAY_BUILD_TEST_UTILS FALSE CACHE BOOL
      "Turn off the build of the Detray tests utilities" )
endif()
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
set( DETRAY_SETUP_THRUST FALSE CACHE BOOL
   "Do not set up Thrust as part of Detray" )
set( DETRAY_SETUP_COVFIE FALSE CACHE BOOL
   "Do not set up covfie as part of Detray" )

# Get it into the current directory.
FetchContent_MakeAvailable( Detray )
