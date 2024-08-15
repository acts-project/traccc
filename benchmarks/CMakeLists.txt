# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( traccc-compiler-options-cpp )

# Set up a common library, shared by all of the tests.
add_library( traccc_benchmarks_common STATIC
    "common/benchmarks/toy_detector_benchmark.hpp" )
target_include_directories( traccc_benchmarks_common
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common )
target_link_libraries( traccc_benchmarks_common
    PUBLIC benchmark::benchmark benchmark::benchmark_main 
    traccc::core traccc::io traccc::simulation detray::core detray::utils 
    vecmem::core Boost::filesystem)

add_subdirectory(cpu)
if( TRACCC_BUILD_CUDA )
    add_subdirectory(cuda)
endif()