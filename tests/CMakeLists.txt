# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( traccc-compiler-options-cpp )

# Set up a common library, shared by all of the tests.
add_library( traccc_tests_common INTERFACE )
target_include_directories( traccc_tests_common
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/common )

# Add all of the test subdirectories.
add_subdirectory( core )
add_subdirectory( cpu )
add_subdirectory( io )

if( TRACCC_BUILD_CUDA )
    add_subdirectory( cuda )
endif()

if( TRACCC_BUILD_KOKKOS )
    add_subdirectory( kokkos )
endif()

if(TRACCC_BUILD_FUTHARK)
    add_subdirectory(futhark)
endif()
