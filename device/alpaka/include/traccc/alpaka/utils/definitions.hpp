/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#ifdef alpaka_ACC_GPU_CUDA_ENABLED
#include "vecmem/utils/cuda/copy.h"
#else
#include "vecmem/utils/copy.hpp"
#endif

namespace traccc::alpaka {

#ifdef alpaka_ACC_GPU_CUDA_ENABLED
#define WARP_SIZE 32
#else
#define WARP_SIZE 4
#endif

using Dim = ::alpaka::DimInt<1>;
using Idx = uint32_t;
using WorkDiv = ::alpaka::WorkDivMembers<Dim, Idx>;

using Acc = ::alpaka::ExampleDefaultAcc<Dim, Idx>;
using Host = ::alpaka::DevCpu;
using Queue = ::alpaka::Queue<Acc, ::alpaka::Blocking>;

}  // namespace traccc::alpaka
