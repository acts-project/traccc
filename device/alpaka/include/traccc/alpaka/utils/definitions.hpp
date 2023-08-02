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
#include <vecmem/utils/cuda/copy.hpp>
#endif

#include <vecmem/utils/copy.hpp>

namespace traccc::alpaka {

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
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

template <typename TAcc>
inline WorkDiv makeWorkDiv(Idx blocks, Idx threadsOrElements) {
    const Idx blocksPerGrid = std::max(Idx{1}, blocks);
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    if constexpr (std::is_same_v<TAcc, ::alpaka::AccGpuCudaRt<Dim, Idx>>) {
        const auto elementsPerThread = Idx{1};
        const Idx threadsPerBlock(threadsOrElements);
        return WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
    } else
#endif
    {
        const auto threadsPerBlock = Idx{1};
        const Idx elementsPerThread(threadsOrElements);
        return WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
    }
}

}  // namespace traccc::alpaka
