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

# option for algebra plugins (ARRAY EIGEN SMATRIX VC VECMEM)
set(TRACCC_ALGEBRA_PLUGIN EIGEN)
if (TRACCC_ALGEBRA_PLUGIN)
  message(STATUS "Building with plugin type: " ${TRACCC_ALGEBRA_PLUGIN})
  set(ALGEBRA_PLUGIN_INCLUDE_${TRACCC_ALGEBRA_PLUGIN} ON)
  add_definitions(-DALGEBRA_PLUGIN_INCLUDE_${TRACCC_ALGEBRA_PLUGIN})
endif()

# option for algebra scalar type 
set(TRACCC_CUSTOM_SCALARTYPE float)
if(TRACCC_CUSTOM_SCALARTYPE)
    message(STATUS "Building with custom scalar type: " ${TRACCC_CUSTOM_SCALARTYPE})
    set(ALGEBRA_PLUGIN_CUSTOM_SCALARTYPE ${TRACCC_CUSTOM_SCALARTYPE})
endif()
    
# Set GPU-related options
if( TRACCC_BUILD_CUDA )
  set( CMAKE_CUDA_STANDARD 17 CACHE STRING "The (CUDA) C++ standard to use" )
  add_subdirectory( device/cuda )
endif()

# Build the traccc code.
add_subdirectory(core)
add_subdirectory(io)
add_subdirectory(examples)
add_subdirectory(plugins)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# Build the necessary externals.
include(traccc-dependencies)
add_subdirectory(extern)
