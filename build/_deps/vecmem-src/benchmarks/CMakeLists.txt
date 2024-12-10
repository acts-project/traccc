# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up Google Benchmark.
option(
   VECMEM_SETUP_GOOGLEBENCHMARK
   "Set up the Google Benchmark targets explicitly"
   TRUE
)
option(
   VECMEM_USE_SYSTEM_GOOGLEBENCHMARK
   "Pick up an existing installation of Google Benchmark from the build environment"
   ${VECMEM_USE_SYSTEM_LIBS}
)
if(VECMEM_SETUP_GOOGLEBENCHMARK)
   if(VECMEM_USE_SYSTEM_GOOGLEBENCHMARK)
      find_package(benchmark REQUIRED)
   else()
      add_subdirectory(googlebenchmark)
   endif()
endif()

# Project include(s).
include( vecmem-compiler-options-cpp )

# Build a common, helper library.
add_library( vecmem_benchmark_common STATIC
   "common/make_jagged_sizes.hpp"
   "common/make_jagged_sizes.cpp"
   "common/make_jagged_vector.hpp"
   "common/make_jagged_vector.cpp" )
target_link_libraries( vecmem_benchmark_common
   PUBLIC vecmem::core )
set_target_properties( vecmem_benchmark_common PROPERTIES
   FOLDER "vecmem/benchmarks" )

# Include the library specific tests.
add_subdirectory(core)
if(VECMEM_BUILD_CUDA_LIBRARY)
   add_subdirectory(cuda)
endif()
if(VECMEM_BUILD_SYCL_LIBRARY)
   add_subdirectory(sycl)
endif()
