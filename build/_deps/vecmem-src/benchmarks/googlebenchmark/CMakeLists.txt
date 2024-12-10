# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.11 )
include( FetchContent )

# Silence various FetchContent warnings up to CMake 3.30.X.
if( POLICY CMP0135 )
   cmake_policy( SET CMP0135 NEW )
endif()
if( POLICY CMP0169 )
   cmake_policy( SET CMP0169 OLD )
endif()

# Tell the user what's happening.
message( STATUS "Building Google Benchmark as part of the VecMem project" )

# Declare where to get googlebenchmark from.
set( VECMEM_BENCHMARK_SOURCE
   "URL;https://github.com/google/benchmark/archive/refs/tags/v1.6.1.tar.gz;URL_MD5;8c33c51f9b7154e6c290df3750081c87"
   CACHE STRING "Source for GoogleBenchmark, when built as part of VecMem" )
mark_as_advanced( VECMEM_BENCHMARK_SOURCE )
FetchContent_Declare( googlebenchmark ${VECMEM_BENCHMARK_SOURCE} )

# Option(s) used in the build of Google Benchmark.
set( BENCHMARK_ENABLE_TESTING OFF CACHE BOOL
   "Enable/disable testing of the benchmark library" )
set( BENCHMARK_ENABLE_WERROR OFF CACHE BOOL
   "Enable/disable using -Werror in the build of Google Benchmark" )

# Get it into the current directory.
FetchContent_Populate( googlebenchmark )
set( CMAKE_FOLDER "vecmem/externals" )
add_subdirectory( "${googlebenchmark_SOURCE_DIR}"
   "${googlebenchmark_BINARY_DIR}" EXCLUDE_FROM_ALL )
