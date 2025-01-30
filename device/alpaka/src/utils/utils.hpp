/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

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

//Get alpaka accelerator
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    using Acc = ::alpaka::AccGpuCudaRt<Dim, Idx>;
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    using Acc = ::alpaka::AccGpuHipRt<Dim, Idx>;
#elif defined(ALPAKA_ACC_CPU_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_CPU)
    using Acc = ::alpaka::AccCpuSycl<Dim, Idx>;
#elif defined(ALPAKA_ACC_GPU_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU)
    using Acc = ::alpaka::AccGpuSyclIntel<Dim, Idx>;
#elif defined(ALPAKA_ACC_FPGA_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_FPGA)
    using Acc = ::alpaka::AccFpgaSyclIntel<Dim, Idx>;
#else
    using Acc = ::alpaka::AccCpuSerial<Dim, Idx>;
#endif

using Host = ::alpaka::DevCpu;
using Queue = ::alpaka::Queue<Acc, ::alpaka::Blocking>;

template <typename TAcc>
consteval std::size_t getWarpSize() {
  if constexpr(::alpaka::accMatchesTags<TAcc, ::alpaka::TagGpuCudaRt, ::alpaka::TagGpuHipRt, ::alpaka::TagGpuSyclIntel>) {
    return 32;
  }
  else {
    return 4;
  }
}

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
