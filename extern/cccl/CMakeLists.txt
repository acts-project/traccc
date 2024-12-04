# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.25 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building CCCL as part of the TRACCC project" )

# Declare where to get Thrust from.
set( TRACCC_CCCL_SOURCE
   "GIT_REPOSITORY;https://github.com/stephenswat/cccl.git;GIT_TAG;build/allow_installing_when_downstream"
   CACHE STRING "Source for CCCL, when built as part of this project" )
mark_as_advanced( TRACCC_CCCL_SOURCE )
# Note that we must not use SYSTEM here. Otherwise nvcc would pick up Thrust
# from its own installation, and not from the version that we are downloading.
FetchContent_Declare( CCCL ${TRACCC_CCCL_SOURCE} )

# There are a few things to configure here... May not all be necessary but
# can't hurt.
set ( CCCL_ENABLE_INSTALL_RULES ON CACHE BOOL "Install CCCL as part of traccc." )
set ( THRUST_ENABLE_INSTALL_RULES ON CACHE BOOL "Install Thrust as part of traccc." )
set ( CUB_ENABLE_INSTALL_RULES ON CACHE BOOL "Install CUB as part of traccc." )
set ( libcudacxx_ENABLE_INSTALL_RULES ON CACHE BOOL "Install libcudacxx as part of traccc." )
set ( CCCL_ENABLE_LIBCUDACXX OFF CACHE BOOL "Disable libcudacxx" )
set ( CCCL_ENABLE_CUB ON CACHE BOOL "Enable CUB" )
set ( CCCL_ENABLE_THRUST ON CACHE BOOL "Enable Thrust" )
set ( CCCL_ENABLE_TESTING OFF CACHE BOOL "Disable CCCL tests" )
set ( CCCL_ENABLE_EXAMPLES OFF CACHE BOOL "Disable CCCL examples" )
set ( CCCL_ENABLE_BENCHMARKS OFF CACHE BOOL "Disable CCCL benchmarks" )
set ( THRUST_ENABLE_TESTING OFF CACHE BOOL "Disable Thrust tests" )
set ( THRUST_ENABLE_EXAMPLES OFF CACHE BOOL "Disable Thrust examples" )
set ( THRUST_ENABLE_HEADER_TESTING OFF CACHE BOOL "Disable Thrust header testing" )
set ( CUB_ENABLE_TESTING OFF CACHE BOOL "Disable CUB tests" )
set ( CUB_ENABLE_EXAMPLES OFF CACHE BOOL "Disable CUB examples" )
set ( CUB_ENABLE_HEADER_TESTING OFF CACHE BOOL "Disable CUB header testing" )

# Get it into the current directory.
FetchContent_MakeAvailable( CCCL )

# Check if <thrust/pair.h> works out of the box.
include( CheckCXXSourceCompiles )
set( CMAKE_REQUIRED_INCLUDES
   "${CCCL_SOURCE_DIR}/thrust"
   "${CCCL_SOURCE_DIR}/libcudacxx/include" )
set( _THRUST_TEST_SOURCE "#include <thrust/pair.h>\nint main() { return 0; }" )
check_cxx_source_compiles( "${_THRUST_TEST_SOURCE}" TRACCC_THRUST_WORKS )
# If not, check if the _LIBCUDACXX_STRING_H_HAS_CONST_OVERLOADS flag makes it
# work.
if( NOT TRACCC_THRUST_WORKS )
   set( CMAKE_REQUIRED_DEFINITIONS -D_LIBCUDACXX_STRING_H_HAS_CONST_OVERLOADS )
   check_cxx_source_compiles( "${_THRUST_TEST_SOURCE}"
      TRACCC_THRUST_WORKS_WITH_PATCH )
   if( TRACCC_THRUST_WORKS_WITH_PATCH )
      target_compile_definitions( _Thrust_Thrust
         INTERFACE _LIBCUDACXX_STRING_H_HAS_CONST_OVERLOADS )
   else()
      message( WARNING "Thrust does not seem to work. The build may fail." )
   endif()
endif()
