# Detray library, part of the ACTS project (R&D line)
#
# (c) 2022-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

message(STATUS "Building detray CUDA benchmarks")

# C++17 support for CUDA requires CMake 3.18.
cmake_minimum_required(VERSION 3.18)

# Enable CUDA as a language.
enable_language(CUDA)

# Set the CUDA build flags.
include(detray-compiler-options-cuda)

# Look for openMP, which is used for the CPU benchmark
find_package(OpenMP)

# make unit tests for multiple algebras
# Currently vc and smatrix is not supported
set(algebras "array")
if(DETRAY_EIGEN_PLUGIN)
    list(APPEND algebras "eigen")
endif()

foreach(algebra ${algebras})
    detray_add_executable(benchmark_cuda_${algebra}
   "benchmark_propagator_cuda_kernel.hpp"
       "benchmark_propagator_cuda.cpp"
       "benchmark_propagator_cuda_kernel.cu"
       LINK_LIBRARIES benchmark::benchmark detray::core detray::algebra_${algebra} vecmem::cuda detray::test_utils
    )

    target_compile_definitions(
        detray_benchmark_cuda_${algebra}
        PRIVATE ${algebra}=${algebra}
    )

    target_compile_options(
        detray_benchmark_cuda_${algebra}
        PRIVATE "-march=native" "-ftree-vectorize"
    )

    if(OpenMP_CXX_FOUND)
        target_link_libraries(
            detray_benchmark_cuda_${algebra}
            PRIVATE OpenMP::OpenMP_CXX
        )
    endif()
endforeach()
