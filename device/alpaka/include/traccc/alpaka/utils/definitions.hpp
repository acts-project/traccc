/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include "/home/wthompson/Work/traccc-build/_deps/vecmem-src/cuda/include/vecmem/utils/cuda/copy.hpp"
#endif

#include "vecmem/utils/copy.hpp"

namespace traccc::alpaka {

using Dim = ::alpaka::DimInt<1>;
using Idx = uint32_t;
using WorkDiv = ::alpaka::WorkDivMembers<Dim, Idx>;

}  // namespace traccc::alpaka
