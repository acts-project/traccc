/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include "../../cuda/src/utils/cuda_error_handling.hpp"

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    int r = RUN_ALL_TESTS();

    int devices = 0;

    cudaGetDeviceCount(&devices);

    if (devices > 0) {
        VECMEM_CUDA_ERROR_CHECK(cudaDeviceReset());
    }

    return r;
}
