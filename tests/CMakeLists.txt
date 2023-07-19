# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( traccc-compiler-options-cpp )

# Set up a common library, shared by all of the tests.
add_library( traccc_tests_common STATIC
    "common/tests/atlas_cuts.hpp"
    "common/tests/cca_test.hpp"
    "common/tests/data_test.hpp"
    "common/tests/kalman_fitting_test.hpp"
    "common/tests/kalman_fitting_test.cpp"
    "common/tests/space_point.hpp" )
target_include_directories( traccc_tests_common
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common )
target_link_libraries( traccc_tests_common
    PUBLIC dfelibs::dfelibs GTest::gtest ActsCore vecmem::core
           detray::core detray::utils covfie::core traccc::core traccc::io )

# Use ROOT in traccc_tests_common, if requested.
if( TRACCC_USE_ROOT )
   find_package( ROOT COMPONENTS Core RIO Hist REQUIRED )
   target_link_libraries( traccc_tests_common
      PRIVATE ROOT::Core ROOT::RIO ROOT::Hist )
   target_compile_definitions( traccc_tests_common
      PRIVATE TRACCC_HAVE_ROOT )
endif()

# Add all of the test subdirectories.
add_subdirectory( core )
add_subdirectory( cpu )
add_subdirectory( io )

if( TRACCC_BUILD_CUDA )
    add_subdirectory( cuda )
endif()

if( TRACCC_BUILD_SYCL )
    add_subdirectory( sycl )
endif()

if( TRACCC_BUILD_KOKKOS )
    add_subdirectory( kokkos )
endif()

if( TRACCC_BUILD_ALPAKA )
    add_subdirectory( alpaka )
endif()

if(TRACCC_BUILD_FUTHARK)
    add_subdirectory(futhark)
endif()
