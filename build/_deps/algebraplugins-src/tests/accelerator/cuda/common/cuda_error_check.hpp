/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// CUDA include(s).
#include <cuda_runtime_api.h>

/// Helper macro used for checking @c cudaError_t type return values.
#define CUDA_ERROR_CHECK(EXP)                                 \
  do {                                                        \
    cudaError_t errorCode = EXP;                              \
    if (errorCode != cudaSuccess) {                           \
      cuda::throw_error(errorCode, #EXP, __FILE__, __LINE__); \
    }                                                         \
  } while (false)

namespace cuda {

/// Function used to print and throw a user-readable error if something breaks
void throw_error(cudaError_t errorCode, const char* expression,
                 const char* file, int line);

}  // namespace cuda
