# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the project.
cmake_minimum_required( VERSION 3.10 )
project( vecmem VERSION 1.13.0 LANGUAGES CXX )

# Standard CMake include(s).
include( GNUInstallDirs )
include( CTest )

# Set the location of the installed CMake files.
set( CMAKE_INSTALL_CMAKEDIR
   "${CMAKE_INSTALL_LIBDIR}/cmake/vecmem-${PROJECT_VERSION}" )

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

# Flags controlling the meta-build system.
option( VECMEM_USE_SYSTEM_LIBS "Use system libraries by default" FALSE )
option( VECMEM_BUILD_TESTING "Build the unit tests of VecMem" TRUE )
option( VECMEM_TEST_UBSAN "Use the undefined behavior sanitizer for the tests"
   TRUE )

# Include the VecMem CMake code.
list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )
include( vecmem-functions )
include( vecmem-options )

# Add the library sub-directories.
add_subdirectory( core )
if( VECMEM_BUILD_CUDA_LIBRARY )
   add_subdirectory( cuda )
endif()
if( VECMEM_BUILD_HIP_LIBRARY )
   add_subdirectory( hip )
endif()
if( VECMEM_BUILD_SYCL_LIBRARY )
   add_subdirectory( sycl )
endif()

# Set up the test(s).
if( BUILD_TESTING AND VECMEM_BUILD_TESTING )
  add_subdirectory( tests )
endif()

# Set up the benchmarks.
option(VECMEM_BUILD_BENCHMARKING "Build the benchmarks of VecMem" OFF)
if(VECMEM_BUILD_BENCHMARKING)
   add_subdirectory(benchmarks)
endif()

# Set up the packaging of the project.
include( vecmem-packaging )

# Set up Doxygen documentation creation for the project, if requested.
if( VECMEM_BUILD_DOCS )

   # This requires / uses Doxygen and doxygen-awesome-css.
   find_package( Doxygen REQUIRED )
   add_subdirectory( cmake/doxygen-awesome-css )

   # Set up how to build the Doxygen documentation.
   set( DOXYGEN_JAVADOC_AUTOBRIEF TRUE )
   set( DOXYGEN_USE_MDFILE_AS_MAINPAGE
      "${CMAKE_CURRENT_SOURCE_DIR}/README.md" )
   set( DOXYGEN_FULL_PATH_NAMES TRUE )
   set( DOXYGEN_INLINE_INHERITED_MEMB TRUE )
   set( DOXYGEN_INHERIT_DOCS TRUE )
   set( DOXYGEN_EXTRACT_PRIV_VIRTUAL TRUE )
   set( DOXYGEN_STRIP_FROM_PATH
      "${CMAKE_CURRENT_SOURCE_DIR}" )
   set( DOXYGEN_STRIP_FROM_INC_PATH
      "${CMAKE_CURRENT_SOURCE_DIR}/core/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/cuda/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/hip/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/sycl/include" )
   set( DOXYGEN_PREDEFINED
      "__cplusplus=201700L" )
   set( DOXYGEN_EXCLUDE
      "${CMAKE_CURRENT_SOURCE_DIR}/tests"
      "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks"
      "${CMAKE_CURRENT_SOURCE_DIR}/build" )
   set( DOXYGEN_GENERATE_TREEVIEW FALSE )
   set( DOXYGEN_DISABLE_INDEX FALSE )
   set( DOXYGEN_FULL_SIDEBAR FALSE )
   set( DOXYGEN_HTML_EXTRA_STYLESHEET
      "${doxygenawesomecss_SOURCE_DIR}/doxygen-awesome.css" )

   # Set up the "vecmem_docs" target.
   doxygen_add_docs( vecmem_docs
      "${CMAKE_CURRENT_SOURCE_DIR}"
      COMMENT "Generating Doxygen pages" )
endif()
