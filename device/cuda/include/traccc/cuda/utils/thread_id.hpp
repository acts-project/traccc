/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cstddef>

#include "traccc/definitions/qualifiers.hpp"

namespace traccc::cuda {
struct thread_id1 {
    TRACCC_DEVICE thread_id1() {}

    std::size_t inline TRACCC_DEVICE getLocalThreadId() const {
        return threadIdx.x;
    }

    std::size_t inline TRACCC_DEVICE getLocalThreadIdX() const {
        return threadIdx.x;
    }

    std::size_t inline TRACCC_DEVICE getGlobalThreadId() const {
        return threadIdx.x + blockIdx.x * blockDim.x;
    }

    std::size_t inline TRACCC_DEVICE getGlobalThreadIdX() const {
        return threadIdx.x + blockIdx.x * blockDim.x;
    }

    std::size_t inline TRACCC_DEVICE getBlockIdX() const { return blockIdx.x; }

    std::size_t inline TRACCC_DEVICE getBlockDimX() const { return blockDim.x; }

    std::size_t inline TRACCC_DEVICE getGridDimX() const { return gridDim.x; }
};
}  // namespace traccc::cuda
