/**
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#if defined(__CUDA_ARCH__) && defined(TRACCC_CUDA_ENABLE_SPILL_TO_SHARED_MEMORY)
#define TRACCC_CUDA_SPILL_TO_SHARED_MEMORY        \
    do {                                          \
        asm(".pragma \"enable_smem_spilling\";"); \
    } while (0)
#else
#define TRACCC_CUDA_SPILL_TO_SHARED_MEMORY \
    do {                                   \
    } while (0)
#endif
