# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Guard against multiple includes.
include_guard( GLOBAL )

# Tell the user what's happening.
message( STATUS "Building detray as part of the traccc project" )

# Declare where to get VecMem from.
FetchContent_Declare( Detray
  GIT_REPOSITORY "https://github.com/acts-project/detray.git"
  GIT_TAG        origin/main)

# Prevent Detray from building its tests and benchmarks
# builds/uses GoogleTest.
set( DETRAY_UNIT_TESTS OFF )
set( DETRAY_BENCHMARKS OFF )
set( DETRAY_CUSTOM_SCALARTYPE ${TRACCC_CUSTOM_SCALARTYPE} )
set( DETRAY_${TRACCC_ALGEBRA_PLUGIN}_PLUGIN ON )
if(TRACC_BUILD_CUDA)
  set( DETRAY_BUILD_CUDA ON )
endif()  

# Get it into the current directory.
FetchContent_Populate( Detray )
add_subdirectory( "${detray_SOURCE_DIR}" "${detray_BINARY_DIR}"
  EXCLUDE_FROM_ALL )
