/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/utils/cuda/stream_wrapper.hpp"

// CUDA include(s).
#include <cuda_runtime_api.h>

namespace vecmem {
namespace cuda {
namespace details {

/**
 * @brief Get current CUDA device number.
 *
 * This function wraps the cudaGetDevice function in a way that returns the
 * device number rather than use a reference argument to write to.
 *
 * Note that calling the function on a machine with no CUDA device does not
 * result in an error, the function just returns 0 in that case.
 */
int get_device();

/// Get concrete @c cudaStream_t object out of our wrapper
cudaStream_t get_stream(const stream_wrapper& stream);

}  // namespace details
}  // namespace cuda
}  // namespace vecmem
