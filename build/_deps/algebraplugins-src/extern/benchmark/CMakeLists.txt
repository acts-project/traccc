# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.11 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building Google Benchmark as part of the Algebra Plugins project" )

# Declare where to get GoogleTest from.
set( ALGEBRA_PLUGINS_BENCHMARK_SOURCE
   "URL;https://github.com/google/benchmark/archive/refs/tags/v1.6.1.tar.gz;URL_MD5;8c33c51f9b7154e6c290df3750081c87"
   CACHE STRING "Source for Google Benchmark, when built as part of this project" )
mark_as_advanced( ALGEBRA_PLUGINS_BENCHMARK_SOURCE )

# Mark the import as a system library on modern CMake versions
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.25.0)
   set(ALGEBRA_PLUGINS_BENCHMARK_SOURCE_FULL "${ALGEBRA_PLUGINS_BENCHMARK_SOURCE};SYSTEM")
else()
   set(ALGEBRA_PLUGINS_BENCHMARK_SOURCE_FULL "${ALGEBRA_PLUGINS_BENCHMARK_SOURCE}")
endif()
mark_as_advanced( ALGEBRA_PLUGINS_BENCHMARK_SOURCE_FULL )

FetchContent_Declare( benchmark ${ALGEBRA_PLUGINS_BENCHMARK_SOURCE_FULL} )

# Options used in the build of Google Benchmark.
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Don't build the Google Benchmark tests")
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Turn off the installation of Google Benchmark")
set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "Don't install the Google Benchmark docs")
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Don't build the unit tests which depend on gtest")
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "Enable/disable using -Werror in the build of Google Benchmark")

# Get it into the current directory.
FetchContent_Populate( benchmark )
add_subdirectory( "${benchmark_SOURCE_DIR}" "${benchmark_BINARY_DIR}"
   EXCLUDE_FROM_ALL )
