/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

namespace traccc::alpaka {

using Dim = ::alpaka::DimInt<1>;
using Idx = uint32_t;
using WorkDiv = ::alpaka::WorkDivMembers<Dim, Idx>;

using Acc = ::alpaka::ExampleDefaultAcc<Dim, Idx>;
using Host = ::alpaka::DevCpu;
using Queue = ::alpaka::Queue<Acc, ::alpaka::NonBlocking>;

template <typename TAcc>
inline WorkDiv makeWorkDiv(Idx blocksPerGrid, Idx threadsOrElements) {
    const Idx blocksPerGrid = std::max(Idx{1}, blocks);
    if constexpr (::alpaka::accMatchesTags<TAcc, ::alpaka::TagGpuCudaRt>) {
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
