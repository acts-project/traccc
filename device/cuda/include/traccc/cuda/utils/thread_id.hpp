/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/definitions/qualifiers.hpp"

namespace traccc::cuda {
struct thread_id1 {
    TRACCC_DEVICE thread_id1() {}

    unsigned int inline TRACCC_DEVICE getLocalThreadId() const {
        return threadIdx.x;
    }

    unsigned int inline TRACCC_DEVICE getLocalThreadIdX() const {
        return threadIdx.x;
    }

    unsigned int inline TRACCC_DEVICE getGlobalThreadId() const {
        return threadIdx.x + blockIdx.x * blockDim.x;
    }

    unsigned int inline TRACCC_DEVICE getGlobalThreadIdX() const {
        return threadIdx.x + blockIdx.x * blockDim.x;
    }

    unsigned int inline TRACCC_DEVICE getBlockIdX() const { return blockIdx.x; }

    unsigned int inline TRACCC_DEVICE getBlockDimX() const {
        return blockDim.x;
    }

    unsigned int inline TRACCC_DEVICE getGridDimX() const { return gridDim.x; }
};
}  // namespace traccc::cuda
