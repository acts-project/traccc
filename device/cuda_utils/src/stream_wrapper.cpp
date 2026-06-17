/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/cuda/utils/stream_wrapper.hpp"

#include "cuda_error_handling.hpp"

// CUDA include(s).
#include <cuda_runtime_api.h>

namespace traccc::cuda {

stream_wrapper::stream_wrapper(void* stream) : m_stream{stream} {}

int stream_wrapper::device() const {

    int device = -1;
    TRACCC_CUDA_ERROR_CHECK(
        cudaStreamGetDevice(static_cast<cudaStream_t>(m_stream), &device));
    return device;
}

void* stream_wrapper::cudaStream() const {

    return m_stream;
}

void stream_wrapper::synchronize() const {

    TRACCC_CUDA_ERROR_CHECK(
        cudaStreamSynchronize(static_cast<cudaStream_t>(m_stream)));
}

}  // namespace traccc::cuda
