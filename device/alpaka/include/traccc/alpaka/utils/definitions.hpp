/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <vecmem/utils/cuda/copy.hpp>
#endif

#include <vecmem/utils/copy.hpp>

namespace traccc::alpaka {

static constexpr std::size_t warpSize =
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    32;
#else
    4;
#endif

}  // namespace traccc::alpaka
