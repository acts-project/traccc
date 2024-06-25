/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
// Set values of maximum resident threads per SM
#if __CUDA_ARCH__ < 500
#pragma message \
    "Very old CUDA architecture, setting maximum resident threads per SM to 1024."
#define CUDA_MAX_RESIDENT_THREADS_PER_SM 1024u
#elif __CUDA_ARCH__ <= 720
#define CUDA_MAX_RESIDENT_THREADS_PER_SM 2048u
#elif __CUDA_ARCH__ <= 750
#define CUDA_MAX_RESIDENT_THREADS_PER_SM 1024u
#elif __CUDA_ARCH__ <= 800
#define CUDA_MAX_RESIDENT_THREADS_PER_SM 2048u
#elif __CUDA_ARCH__ <= 890
#define CUDA_MAX_RESIDENT_THREADS_PER_SM 1536u
#elif __CUDA_ARCH__ <= 900
#define CUDA_MAX_RESIDENT_THREADS_PER_SM 2048u
#pragma message \
    "Unknown CUDA architecture, setting maximum resident threads per SM to 1024."
#define CUDA_MAX_RESIDENT_THREADS_PER_SM 1024u
#endif

// Set values of maximum resident blocks per SM
#if __CUDA_ARCH__ < 500
#pragma message \
    "Very old CUDA architecture, setting maximum resident blocks per SM to 16."
#define CUDA_MAX_RESIDENT_BLOCKS_PER_SM 16u
#elif __CUDA_ARCH__ <= 720
#define CUDA_MAX_RESIDENT_BLOCKS_PER_SM 32u
#elif __CUDA_ARCH__ <= 750
#define CUDA_MAX_RESIDENT_BLOCKS_PER_SM 16u
#elif __CUDA_ARCH__ <= 800
#define CUDA_MAX_RESIDENT_BLOCKS_PER_SM 32u
#elif __CUDA_ARCH__ <= 870
#define CUDA_MAX_RESIDENT_BLOCKS_PER_SM 16u
#elif __CUDA_ARCH__ <= 890
#define CUDA_MAX_RESIDENT_BLOCKS_PER_SM 24u
#elif __CUDA_ARCH__ <= 900
#define CUDA_MAX_RESIDENT_BLOCKS_PER_SM 32u
#else
#pragma message \
    "Very old CUDA architecture, setting maximum resident blocks per SM to 16."
#define CUDA_MAX_RESIDENT_BLOCKS_PER_SM 16u
#endif

// Set values of maximum threadss per block
#define CUDA_MAX_THREADS_PER_BLOCK 1024u
#endif
