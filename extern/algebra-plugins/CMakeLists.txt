# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.25 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building Algebra Plugins as part of the TRACCC project" )

# Declare where to get Algebra Plugins from.
set( TRACCC_ALGEBRA_PLUGINS_SOURCE
   "URL;https://github.com/acts-project/algebra-plugins/archive/refs/tags/v0.26.2.tar.gz;URL_MD5;607bd0b8beffcf7981b10c03f237e550"
   CACHE STRING "Source for Algebra Plugins, when built as part of this project" )
mark_as_advanced( TRACCC_ALGEBRA_PLUGINS_SOURCE )
FetchContent_Declare( AlgebraPlugins SYSTEM ${TRACCC_ALGEBRA_PLUGINS_SOURCE} )

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

# Get it into the current directory.
FetchContent_MakeAvailable( AlgebraPlugins )
