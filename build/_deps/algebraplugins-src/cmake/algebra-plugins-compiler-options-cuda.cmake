# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# FindCUDAToolkit needs at least CMake 3.17.
cmake_minimum_required( VERSION 3.17 )

# Include the helper function(s).
include( algebra-plugins-functions )

# Figure out the properties of CUDA being used.
find_package( CUDAToolkit REQUIRED )

# Set the architecture to build code for.
set( CMAKE_CUDA_ARCHITECTURES "52" CACHE STRING
   "CUDA architectures to build device code for" )

# Turn on the correct setting for the __cplusplus macro with MSVC.
if( "${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC" )
   algebra_add_flag( CMAKE_CUDA_FLAGS "-Xcompiler /Zc:__cplusplus" )
endif()

# Make CUDA generate debug symbols for the device code as well in a debug
# build.
if( "${CMAKE_CUDA_COMPILER_ID}" MATCHES "NVIDIA" )
   algebra_add_flag( CMAKE_CUDA_FLAGS_DEBUG "-G" )
endif()

# Fail on warnings, if asked for that behaviour.
if( ALGEBRA_PLUGINS_FAIL_ON_WARNINGS )
   if( ( "${CUDAToolkit_VERSION}" VERSION_GREATER_EQUAL "10.2" ) AND
       ( "${CMAKE_CUDA_COMPILER_ID}" MATCHES "NVIDIA" ) )
      algebra_add_flag( CMAKE_CUDA_FLAGS "-Werror all-warnings" )
   elseif( "${CMAKE_CUDA_COMPILER_ID}" MATCHES "Clang" )
      algebra_add_flag( CMAKE_CUDA_FLAGS "-Werror" )
   endif()
endif()
