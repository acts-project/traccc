/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "cuda_wrappers.hpp"

#include "cuda_error_handling.hpp"

// CUDA include(s).
#include <cuda_runtime_api.h>

// System include(s).
#include <cassert>

namespace vecmem::cuda::details {

int get_device() {

    int d = 0;
    VECMEM_CUDA_ERROR_IGNORE(cudaGetDevice(&d));
    return d;
}

cudaStream_t get_stream(const stream_wrapper& stream) {

    assert(stream.stream() != nullptr);
    return reinterpret_cast<cudaStream_t>(stream.stream());
}

}  // namespace vecmem::cuda::details
