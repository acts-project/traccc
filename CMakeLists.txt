cmake_minimum_required(VERSION 3.9)

project(traccc VERSION 0.0 LANGUAGES CXX)

enable_testing()

option(TRACCC_UNIT_TESTS "Enable unit tests" On)
option(TRACCC_BENCHMARKS "Enable benchmark tests" On)
option(TRACCC_BUILD_CUDA "Build CUDA" ON)

set(CMAKE_CXX_STANDARD 17)
if(TRACCC_BUILD_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17 CACHE STRING "CUDA C++ standard to use")
endif()

set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX}/include)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/lib)
set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

set(TF_BUILD_EXAMPLES OFF)
set(TF_BUILD_TESTS OFF)
set(INSTALL_GTEST OFF)
set(BENCHMARK_ENABLE_INSTALL OFF)
set(BENCHMARK_ENABLE_TESTING OFF)

add_subdirectory(extern)

add_subdirectory(core)
add_subdirectory(io)
add_subdirectory(examples)
add_subdirectory(tests)
add_subdirectory(algorithms)
