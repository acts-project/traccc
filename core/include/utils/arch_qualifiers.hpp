/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Define a qualifier for cuda
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#define __CUDA_HOST_DEVICE__ inline __host__ __device__
#else
#define __CUDA_HOST_DEVICE__
#endif
