# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the project.
cmake_minimum_required( VERSION 3.9 )
project( traccc VERSION 0.2.0 LANGUAGES CXX )

# Set up the used C++ standard(s).
set( CMAKE_CXX_STANDARD 17 CACHE STRING "The (host) C++ standard to use" )
set( CMAKE_CXX_EXTENSIONS FALSE CACHE BOOL "Disable (host) C++ extensions" )
set( CMAKE_CUDA_STANDARD 17 CACHE STRING "The (CUDA) C++ standard to use" )
set( CMAKE_CUDA_EXTENSIONS FALSE CACHE BOOL "Disable (CUDA) C++ extensions" )
set( CMAKE_SYCL_STANDARD 17 CACHE STRING "The (SYCL) C++ standard to use" )

# Standard CMake include(s).
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
include( traccc-functions )

# Check if CUDA is available.
include( CheckLanguage )
check_language( CUDA )
set( TRACCC_BUILD_CUDA_DEFAULT FALSE )
if( CMAKE_CUDA_COMPILER )
   set( TRACCC_BUILD_CUDA_DEFAULT TRUE )
endif()

# Flags controlling which parts of traccc to build.
option( TRACCC_BUILD_CUDA "Build the CUDA sources included in traccc"
   ${TRACCC_BUILD_CUDA_DEFAULT} )
option( TRACCC_BUILD_SYCL "Build the SYCL sources included in traccc" FALSE )
option( TRACCC_BUILD_FUTHARK "Build the Futhark sources included in traccc"
   FALSE )
option( TRACCC_BUILD_KOKKOS "Build the Kokkos sources included in traccc"
   FALSE )
option( TRACCC_BUILD_TESTING "Build the (unit) tests of traccc" TRUE )
option( TRACCC_BUILD_EXAMPLES "Build the examples of traccc" TRUE )

# Flags controlling the meta-build system.
option( TRACCC_USE_SYSTEM_LIBS "Use system libraries be default" FALSE )

# Clean up.
unset( TRACCC_BUILD_CUDA_DEFAULT )

# Set up VecMem.
option( TRACCC_SETUP_VECMEM
   "Set up the VecMem target(s) explicitly" TRUE )
option( TRACCC_USE_SYSTEM_VECMEM
   "Pick up an existing installation of VecMem from the build environment"
   ${TRACCC_USE_SYSTEM_LIBS} )
if( TRACCC_SETUP_VECMEM )
   if( TRACCC_USE_SYSTEM_VECMEM )
      find_package( vecmem 0.7.0 REQUIRED COMPONENTS LANGUAGE )
   else()
      add_subdirectory( extern/vecmem )
      # Make the "VecMem language code" available for the whole project.
      include( "${VECMEM_LANGUAGE_DIR}/vecmem-check-language.cmake" )
   endif()
endif()

# Set up Eigen3.
option( TRACCC_SETUP_EIGEN3
   "Set up the Eigen3 target(s) explicitly" TRUE )
option( TRACCC_USE_SYSTEM_EIGEN3
   "Pick up an existing installation of Eigen3 from the build environment"
   ${TRACCC_USE_SYSTEM_LIBS} )
if( TRACCC_SETUP_EIGEN3 )
   if( TRACCC_USE_SYSTEM_EIGEN3 )
      find_package( Eigen3 REQUIRED )
   else()
      add_subdirectory( extern/eigen3 )
   endif()
endif()

# Set up Thrust.
option( TRACCC_SETUP_THRUST
   "Set up the Thrust target(s) explicitly" TRUE )
option( TRACCC_USE_SYSTEM_THRUST
   "Pick up an existing installation of Thrust from the build environment"
   ${TRACCC_USE_SYSTEM_LIBS} )
if( TRACCC_SETUP_THRUST )
   if( TRACCC_USE_SYSTEM_THRUST )
      find_package( Thrust REQUIRED )
   else()
      add_subdirectory( extern/thrust )
   endif()
endif()
# Set up an IMPORTED library on top of the Thrust library/libraries. One that
# the TRACCC/Detray code could depend on publicly.
set( TRACCC_THRUST_OPTIONS "" CACHE STRING
   "Extra options for configuring how Thrust should be used" )
mark_as_advanced( TRACCC_THRUST_OPTIONS )
thrust_create_target( traccc::Thrust ${TRACCC_THRUST_OPTIONS} )

# Set up Kokkos.
option( TRACCC_SETUP_KOKKOS
   "Set up the Kokkos library" TRUE )
option( TRACCC_USE_SYSTEM_KOKKOS
   "Pick up an existing installation of Kokkos from the build environment"
   ${TRACCC_USE_SYSTEM_LIBS} )
if( TRACCC_SETUP_KOKKOS )
   if( TRACCC_USE_SYSTEM_KOKKOS )
      find_package( Kokkos REQUIRED )
   else()
      add_subdirectory( extern/kokkos )
   endif()
endif()

# Set up Algebra Plugins.
option( TRACCC_SETUP_ALGEBRA_PLUGINS
   "Set up the Algebra Plugins target(s) explicitly" TRUE )
option( TRACCC_USE_SYSTEM_ALGEBRA_PLUGINS
   "Pick up an existing installation of Algebra Plugins from the build environment"
   ${TRACCC_USE_SYSTEM_LIBS} )
if( TRACCC_SETUP_ALGEBRA_PLUGINS )
   if( TRACCC_USE_SYSTEM_ALGEBRA_PLUGINS )
      find_package( algebra-plugins REQUIRED )
   else()
      add_subdirectory( extern/algebra-plugins )
   endif()
endif()

# Set up dfelibs.
option( TRACCC_SETUP_DFELIBS
   "Set up the dfelibs target(s) explicitly" TRUE )
option( TRACCC_USE_SYSTEM_DFELIBS
   "Pick up an existing installation of dfelibs from the build environment"
   ${TRACCC_USE_SYSTEM_LIBS} )
if( TRACCC_SETUP_DFELIBS )
   if( TRACCC_USE_SYSTEM_DFELIBS )
      find_package( dfelibs REQUIRED )
   else()
      add_subdirectory( extern/dfelibs )
   endif()
endif()

# Set up Detray.
option( TRACCC_SETUP_DETRAY
   "Set up the Detray target(s) explicitly" TRUE )
option( TRACCC_USE_SYSTEM_DETRAY
   "Pick up an existing installation of Detray from the build environment"
   ${TRACCC_USE_SYSTEM_LIBS} )
if( TRACCC_SETUP_DETRAY )
   if( TRACCC_USE_SYSTEM_DETRAY )
      find_package( detray REQUIRED )
   else()
      add_subdirectory( extern/detray )
   endif()
endif()

# Set up Acts.
option( TRACCC_SETUP_ACTS
   "Set up the Acts target(s) explicitly" TRUE )
option( TRACCC_USE_SYSTEM_ACTS
   "Pick up an existing installation of Acts from the build environment"
   ${TRACCC_USE_SYSTEM_LIBS} )
if( TRACCC_SETUP_ACTS )
   if( TRACCC_USE_SYSTEM_ACTS )
      find_package( Acts REQUIRED )
   else()
      add_subdirectory( extern/acts )
   endif()
endif()

# Set up GoogleTest.
option( TRACCC_SETUP_GOOGLETEST
   "Set up the GoogleTest target(s) explicitly" TRUE )
option( TRACCC_USE_SYSTEM_GOOGLETEST
   "Pick up an existing installation of GoogleTest from the build environment"
   ${TRACCC_USE_SYSTEM_LIBS} )
if( TRACCC_SETUP_GOOGLETEST )
   if( TRACCC_USE_SYSTEM_GOOGLETEST )
      find_package( GTest REQUIRED )
   else()
      add_subdirectory( extern/googletest )
   endif()
endif()

# option for algebra plugins (ARRAY EIGEN SMATRIX VC VECMEM)
set(TRACCC_ALGEBRA_PLUGINS ARRAY CACHE STRING "Algebra plugin to use in the build")
message(STATUS "Building with plugin type: " ${TRACCC_ALGEBRA_PLUGINS})
add_definitions(-DALGEBRA_PLUGINS_INCLUDE_${TRACCC_ALGEBRA_PLUGINS})

# Build the traccc code.
add_subdirectory( core )
add_subdirectory( device/common )
if( TRACCC_BUILD_CUDA )
   add_subdirectory( device/cuda )
endif()
if( TRACCC_BUILD_SYCL )
   add_subdirectory( device/sycl )
endif()
add_subdirectory( io )
add_subdirectory( plugins )

if ( TRACCC_BUILD_EXAMPLES )
   # Find Boost.
   find_package( Boost REQUIRED COMPONENTS program_options)
   # Find ROOT.
   find_package( ROOT COMPONENTS Core RIO Tree Hist )
   add_subdirectory( performance )
   add_subdirectory( examples )
endif()

# Set up the test(s).
include( CTest )
if( BUILD_TESTING AND TRACCC_BUILD_TESTING )
   add_subdirectory( tests )
endif()

if(TRACCC_BUILD_FUTHARK)
   add_subdirectory(device/futhark)
endif()
