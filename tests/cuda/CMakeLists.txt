# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

enable_language(CUDA)

include(traccc-compiler-options-cuda)

find_package(CUDAToolkit REQUIRED)

add_library(
    traccc_tests_cuda_main
    STATIC
    cuda_main.cpp
)

target_link_libraries(
    traccc_tests_cuda_main
    PRIVATE
    CUDA::cudart
    GTest::gtest
)

traccc_add_test(
    cuda

    # Define the sources for the test.
    test_basic.cu
    test_cca.cpp
    test_copy_algs.cpp
    test_graphs.cu
    test_graph_concepts.cpp

    LINK_LIBRARIES
    CUDA::cudart
    GTest::gtest
    vecmem::cuda
    traccc::core
    traccc::device_common
    traccc::cuda
    traccc::io
    traccc_tests_cuda_main
    traccc_tests_common
)

target_compile_features(traccc_test_cuda PRIVATE cxx_std_20)
