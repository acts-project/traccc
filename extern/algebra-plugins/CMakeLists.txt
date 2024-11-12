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
message( STATUS "Building Algebra Plugins as part of the TRACCC project" )

# Declare where to get Algebra Plugins from.
set( TRACCC_ALGEBRA_PLUGINS_SOURCE
   "URL;https://github.com/acts-project/algebra-plugins/archive/refs/tags/v0.25.0.tar.gz;URL_MD5;8201744f8b2a8ff7fbcf21bd1b4549b9"
   CACHE STRING "Source for Algebra Plugins, when built as part of this project" )

mark_as_advanced( TRACCC_ALGEBRA_PLUGINS_SOURCE )
FetchContent_Declare( AlgebraPlugins ${TRACCC_ALGEBRA_PLUGINS_SOURCE} )

# Options used in the build of Algebra Plugins.
set( ALGEBRA_PLUGINS_BUILD_TESTING FALSE CACHE BOOL
   "Turn off the build of the Algebra Plugins unit tests" )
set( ALGEBRA_PLUGINS_INCLUDE_EIGEN TRUE CACHE BOOL
   "Turn on the build of algebra::eigen" )
set( ALGEBRA_PLUGINS_INCLUDE_VECMEM TRUE CACHE BOOL
   "Turn on the build of algebra::vecmem_array" )

set( ALGEBRA_PLUGINS_SETUP_EIGEN3 FALSE CACHE BOOL
   "Do not set up Eigen4=3 in Algebra Plugins" )
set( ALGEBRA_PLUGINS_SETUP_VECMEM FALSE CACHE BOOL
   "Do not set up VecMem in Algebra Plugins" )
set( ALGEBRA_PLUGINS_SETUP_GOOGLETEST FALSE CACHE BOOL
   "Do not set up GoogleTest in Algebra Plugins" )
set( ALGEBRA_PLUGINS_SETUP_BENCHMARK FALSE CACHE BOOL
   "Do not set up GoogleTest in Algebra Plugins" )

# Disable a pesky warning coming from the Vc build.
include( CheckCXXCompilerFlag )
check_cxx_compiler_flag( "-Wno-deprecated-enum-enum-conversion"
   TRACCC_HAS_NO_DEPRECATED_ENUM_ENUM_CONVERSION )
if( TRACCC_HAS_NO_DEPRECATED_ENUM_ENUM_CONVERSION )
   traccc_add_flag( CMAKE_CXX_FLAGS "-Wno-deprecated-enum-enum-conversion" )
endif()

# Get it into the current directory.
FetchContent_MakeAvailable( AlgebraPlugins )
