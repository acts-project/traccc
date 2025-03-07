# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Look for openMP, which is used for the CPU benchmark
find_package(OpenMP)

# Build the benchmark executable.
traccc_add_executable(benchmark_cpu
    "toy_detector_cpu.cpp"
    LINK_LIBRARIES benchmark::benchmark benchmark::benchmark_main
    traccc::core traccc_benchmarks_common
    detray::core detray::detectors vecmem::core)

if(OpenMP_CXX_FOUND)
    target_link_libraries(traccc_benchmark_cpu PRIVATE OpenMP::OpenMP_CXX)
endif()
