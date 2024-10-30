/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <vecmem/utils/cuda/copy.hpp>
#endif

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
#include <vecmem/utils/hip/copy.hpp>
#endif

#include <vecmem/utils/copy.hpp>

namespace traccc::alpaka {

using Dim = ::alpaka::DimInt<1>;
using Idx = uint32_t;
using WorkDiv = ::alpaka::WorkDivMembers<Dim, Idx>;

using Acc = ::alpaka::ExampleDefaultAcc<Dim, Idx>;
using Host = ::alpaka::DevCpu;
using Queue = ::alpaka::Queue<Acc, ::alpaka::Blocking>;

static constexpr std::size_t warpSize =
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    32;
#else
    4;
#endif

template <typename TAcc>
inline WorkDiv makeWorkDiv(Idx blocks, Idx threadsOrElements) {
    const Idx blocksPerGrid = std::max(Idx{1}, blocks);
    if constexpr (::alpaka::isMultiThreadAcc<TAcc>) {
        const Idx threadsPerBlock(threadsOrElements);
        const Idx elementsPerThread = Idx{1};
        return WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
    } else {
        const Idx threadsPerBlock = Idx{1};
        const Idx elementsPerThread(threadsOrElements);
        return WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
    }
}

}  // namespace traccc::alpaka
