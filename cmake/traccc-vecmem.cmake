# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Guard against multiple includes.
include_guard( GLOBAL )

# Look for VecMem, quietly first time around. Since if it is not found (which is
# likely), it prints multiple lines of warnings.
find_package( vecmem QUIET )

# If it was found, then we're finished.
if( vecmem_FOUND )
   # Call find_package again, just to nicely print where it is picked up from.
   find_package( vecmem )
   return()
endif()

# CMake include(s).
cmake_minimum_required( VERSION 3.11 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building VecMem as part of the traccc project" )

# Declare where to get VecMem from.
FetchContent_Declare( VecMem
   GIT_REPOSITORY "https://github.com/acts-project/vecmem.git"
   GIT_TAG "v0.2.0" )

# Prevent VecMem from building its tests. As it would interfere with how traccc
# builds/uses GoogleTest.
set( BUILD_TESTING FALSE )

# Get it into the current directory.
FetchContent_Populate( VecMem )
add_subdirectory( "${vecmem_SOURCE_DIR}" "${vecmem_BINARY_DIR}"
   EXCLUDE_FROM_ALL )
