# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.11 )
include( FetchContent )

# Silence FetchContent warnings with CMake >=3.24.
if( POLICY CMP0135 )
   cmake_policy( SET CMP0135 NEW )
endif()

# Tell the user what's happening.
message( STATUS
   "Downloading doxygen-awesome-css as part of the VecMem project" )

# Declare where to get doxygen-awesome-css from.
FetchContent_Declare( DoxygenAwesomeCss
   URL "https://github.com/jothepro/doxygen-awesome-css/archive/refs/tags/v2.3.3.tar.gz"
   URL_MD5 "9b7a51265777e340561f6254421c107d" )

# Make it awailable in the build directory.
set( CMAKE_FOLDER "vecmem/externals" )
FetchContent_MakeAvailable( DoxygenAwesomeCss )
set( doxygenawesomecss_SOURCE_DIR "${doxygenawesomecss_SOURCE_DIR}"
   PARENT_SCOPE )
