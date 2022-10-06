# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.16 )
include( FetchContent )

# Silence FetchContent warnings with CMake >=3.24.
if( POLICY CMP0135 )
   cmake_policy( SET CMP0135 NEW )
endif()

# Tell the user what's happening.
message( STATUS "Building Kokkos as part of the TRACCC project" )

# Declare where to get Kokkos from.
set( TRACCC_KOKKOS_SOURCE
   "URL;https://github.com/kokkos/kokkos/archive/refs/tags/3.7.00.tar.gz;URL_MD5;84991eca9f066383abe119a5bc7a11c4"
   CACHE STRING "Source for Kokkos, when built as part of this project" )
mark_as_advanced( TRACCC_KOKKOS_SOURCE )
FetchContent_Declare( Kokkos ${TRACCC_KOKKOS_SOURCE} )

# Default options for the Kokkos build.
set( Kokkos_ENABLE_SERIAL TRUE CACHE BOOL
   "Enable the serial backend of Kokkos" )

# Get it into the current directory.
FetchContent_MakeAvailable( Kokkos )
