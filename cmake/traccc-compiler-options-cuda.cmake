# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# FindCUDAToolkit needs at least CMake 3.17, and C++17 support
# (set in the project's main CMakeLists.txt file) needs CMake 3.18.
cmake_minimum_required( VERSION 3.18 )

# Include the helper function(s).
include( traccc-functions )

# Figure out the properties of CUDA being used.
find_package( CUDAToolkit REQUIRED )

# Turn on the correct setting for the __cplusplus macro with MSVC.
if( "${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC" )
   traccc_add_flag( CMAKE_CUDA_FLAGS "-Xcompiler /Zc:__cplusplus" )
endif()

# Set the CUDA architecture to build code for.
set( CMAKE_CUDA_ARCHITECTURES "52" CACHE STRING
   "CUDA architectures to build device code for" )

# Allow to use functions in device code that are constexpr, even if they are
# not marked with __device__.
traccc_add_flag( CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr" )

# Make CUDA generate debug symbols for the device code as well in a debug
# build.
traccc_add_flag( CMAKE_CUDA_FLAGS_DEBUG "-G --keep -src-in-ptx" )

# Ensure that line information is embedded in debugging builds so that
# profilers have access to line data.
traccc_add_flag( CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-lineinfo" )

# Fail on warnings, if asked for that behaviour.
if( TRACCC_FAIL_ON_WARNINGS )
   if( ( "${CUDAToolkit_VERSION}" VERSION_GREATER_EQUAL "10.2" ) AND
       ( "${CMAKE_CUDA_COMPILER_ID}" MATCHES "NVIDIA" ) )
      traccc_add_flag( CMAKE_CUDA_FLAGS "-Werror all-warnings" )
   elseif( "${CMAKE_CUDA_COMPILER_ID}" MATCHES "Clang" )
      traccc_add_flag( CMAKE_CUDA_FLAGS "-Werror" )
   endif()
endif()
