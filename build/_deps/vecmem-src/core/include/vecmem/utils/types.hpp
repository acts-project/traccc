/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

/// Macro for declaring a device function
#if defined(__CUDACC__) || defined(__HIP__)
#define VECMEM_DEVICE __device__
#else
#define VECMEM_DEVICE
#endif  // CUDA or HIP

/// Macro for declaring a host function
#if defined(__CUDACC__) || defined(__HIP__)
#define VECMEM_HOST __host__
#else
#define VECMEM_HOST
#endif  // CUDA or HIP

/// Macro for declaring a host+device function
#if defined(__CUDACC__) || defined(__HIP__)
#define VECMEM_HOST_AND_DEVICE __host__ __device__
#else
#define VECMEM_HOST_AND_DEVICE
#endif  // CUDA or HIP
