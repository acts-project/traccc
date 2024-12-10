# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.14 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building Vc as part of the Algebra Plugins project" )

# Declare where to get Vc from.
set( ALGEBRA_PLUGINS_VC_SOURCE
   "URL;https://github.com/VcDevel/Vc/archive/refs/tags/1.4.3.tar.gz;URL_MD5;02d33a9d560a14bb23204be235713f65"
   CACHE STRING "Source for Vc, when built as part of this project" )
mark_as_advanced( ALGEBRA_PLUGINS_VC_SOURCE )

# Mark the import as a system library on modern CMake versions
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.25.0)
   set(ALGEBRA_PLUGINS_VC_SOURCE_FULL "${ALGEBRA_PLUGINS_VC_SOURCE};SYSTEM")
else()
   set(ALGEBRA_PLUGINS_VC_SOURCE_FULL "${ALGEBRA_PLUGINS_VC_SOURCE}")
endif()
mark_as_advanced( ALGEBRA_PLUGINS_VC_SOURCE_FULL )

FetchContent_Declare( Vc ${ALGEBRA_PLUGINS_VC_SOURCE_FULL} )

# Turn off the unit tests for Vc.
if( DEFINED CACHE{BUILD_TESTING} )
   set( _buildTestingValue ${BUILD_TESTING} )
endif()
set( BUILD_TESTING FALSE CACHE INTERNAL "Forceful setting of BUILD_TESTING" )

# Get it into the current directory.
FetchContent_MakeAvailable( Vc )

# Reset the BUILD_TESTING variable.
if( DEFINED _buildTestingValue )
   set( BUILD_TESTING ${_buildTestingValue} CACHE BOOL "Turn tests on/off"
      FORCE )
   unset( _buildTestingValue )
else()
   unset( BUILD_TESTING CACHE )
endif()

# Treat the Vc headers as "system headers", to avoid getting warnings from them.
target_include_directories( Vc
   SYSTEM INTERFACE $<TARGET_PROPERTY:Vc,INCLUDE_DIRECTORIES> )

# Disable a warning for GCC which the Vc build turns on explicitly for that
# compiler, just to then go and trigger it... :-/
if( "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" )
   target_compile_options( Vc PRIVATE "-Wno-old-style-cast" )
endif()

# Set up an alias for the Vc target, with the same name that it will have
# when "finding it".
add_library( Vc::Vc ALIAS Vc )
