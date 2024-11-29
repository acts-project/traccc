# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.25 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building Kokkos as part of the TRACCC project" )

# Declare where to get Kokkos from.
set( TRACCC_KOKKOS_SOURCE
   "URL;https://github.com/kokkos/kokkos/archive/refs/tags/4.3.01.tar.gz;URL_MD5;243de871b3dc2cf3990c1c404032df83"
   CACHE STRING "Source for Kokkos, when built as part of this project" )
mark_as_advanced( TRACCC_KOKKOS_SOURCE )
FetchContent_Declare( Kokkos SYSTEM ${TRACCC_KOKKOS_SOURCE} )

# Default options for the Kokkos build.
set( Kokkos_ENABLE_SERIAL TRUE CACHE BOOL
   "Enable the serial backend of Kokkos" )
set( BUILD_SHARED_LIBS TRUE CACHE BOOL
   "Enable building of shared libs in Kokkos" )

# Get it into the current directory.
FetchContent_MakeAvailable( Kokkos )
