# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the project.
cmake_minimum_required(VERSION 3.9)
project(traccc VERSION 0.0 LANGUAGES CXX)

# Standard CMake include(s).
include( CTest )
include( GNUInstallDirs )

# options
option(TRACCC_ARRAY_PLUGIN "Build std::array math plugin" On)
option(TRACCC_EIGEN_PLUGIN "Build Eigen math plugin" Off)
option(TRACCC_SMATRIX_PLUGIN "Build ROOT/SMatrix math plugin" Off)
option(TRACCC_VC_ALGEBRA_PLUGIN "Build vc based simd algebra math plugin" Off)

if(TRACCC_CUSTOM_SCALARTYPE)
    message(STATUS "Building with custom scalar type: " ${TRACCC_CUSTOM_SCALARTYPE})
    set(ALGEBRA_PLUGIN_CUSTOM_SCALARTYPE ${TRACCC_CUSTOM_SCALARTYPE})
endif()

if (TRACCC_ARRAY_PLUGIN)
    set(ALGEBRA_PLUGIN_INCLUDE_ARRAY ON)
endif()

if (TRACCC_EIGEN_PLUGIN)
    set(ALGEBRA_PLUGIN_INCLUDE_EIGEN ON)
    find_package(Eigen3 REQUIRED)
    include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})
endif()

if (TRACCC_SMATRIX_PLUGIN)
    set(ALGEBRA_PLUGIN_INCLUDE_SMATRIX ON)
    find_package(ROOT REQUIRED COMPONENTS MathCore)
    include_directories(SYSTEM ${ROOT_INCLUDE_DIRS})
endif()

if (TRACCC_VC_ALGEBRA_PLUGIN)
    set(ALGEBRA_PLUGIN_INCLUDE_VC ON)
endif()


# Explicitly set the output directory for the binaries. Such that if this
# project is included by another project, the main project's configuration would
# win out.
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY
   "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}" CACHE PATH
   "Directory for the built binaries" )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY
   "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH
   "Directory for the built libraries" )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY
   "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH
   "Directory for the built static libraries" )

# Include the traccc CMake code.
list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )
include( traccc-compiler-options )

# Build the traccc code.
add_subdirectory(core)
add_subdirectory(io)
add_subdirectory(plugins)
add_subdirectory(examples)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# Build the necessary externals.
add_subdirectory(extern)
include(traccc-dependencies)
