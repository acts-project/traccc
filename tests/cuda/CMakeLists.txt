# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

include(traccc-compiler-options-cuda)

enable_language(CUDA)

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

    LINK_LIBRARIES
    GTest::gtest
    traccc_tests_cuda_main
)
