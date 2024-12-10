# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2022-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( vecmem-compiler-options-cpp )
include( vecmem-compiler-options-sycl )

# Set up the benchmark(s) for the SYCL library.
add_executable( vecmem_benchmark_sycl
    "benchmark_sycl.cpp"
    "benchmark_copy.cpp" )
target_link_libraries( vecmem_benchmark_sycl
    PRIVATE vecmem::sycl vecmem_benchmark_common
            benchmark::benchmark benchmark::benchmark_main
)
set_target_properties( vecmem_benchmark_sycl PROPERTIES
   FOLDER "vecmem/benchmarks" )
