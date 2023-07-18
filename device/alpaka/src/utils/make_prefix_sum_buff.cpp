/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/alpaka/utils/definitions.hpp"

// Project include(s).
#include "traccc/device/make_prefix_sum_buffer.hpp"
#include "traccc/alpaka/utils/make_prefix_sum_buff.hpp"

namespace traccc::alpaka {

struct PrefixSumBuffKernel {
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(
        Acc const& acc,
        vecmem::data::vector_view<const device::prefix_sum_size_t> sizes_view,
        vecmem::data::vector_view<device::prefix_sum_element_t> ps_view
    ) const {
        auto threadIdx = ::alpaka::getIdx<::alpaka::Block, ::alpaka::Threads>(acc)[0u];
        auto blockDim = ::alpaka::getWorkDiv<::alpaka::Block, ::alpaka::Threads>(acc)[0u];
        auto blockIdx = ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Blocks>(acc)[0u];

        device::fill_prefix_sum(threadIdx + blockIdx * blockDim, sizes_view, ps_view);
    }
};

vecmem::data::vector_buffer<device::prefix_sum_element_t> make_prefix_sum_buff(
    const std::vector<device::prefix_sum_size_t>& sizes, vecmem::copy& copy,
    const traccc::memory_resource& mr, Queue& queue) {

    const device::prefix_sum_buffer_t make_sum_result =
        device::make_prefix_sum_buffer(sizes, copy, mr);
    const vecmem::data::vector_view<const device::prefix_sum_size_t>
        sizes_sum_view = make_sum_result.view;
    const unsigned int totalSize = make_sum_result.totalSize;

    // Create buffer and view objects for prefix sum vector
    vecmem::data::vector_buffer<device::prefix_sum_element_t> prefix_sum_buff(
        totalSize, mr.main);
    copy.setup(prefix_sum_buff);
    auto data_prefix_sum_buff = vecmem::get_data(prefix_sum_buff);

    // Setup Alpaka
    auto const deviceProperties = ::alpaka::getAccDevProps<Acc>(::alpaka::getDevByIdx<Acc>(0u));
    auto const maxThreadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];

    // Fixed number of threads per block.
    auto const threadsPerBlock = maxThreadsPerBlock;
    auto const blocksPerGrid = (totalSize + threadsPerBlock - 1) / threadsPerBlock;
    auto const elementsPerThread = 1u;
    auto workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};

    ::alpaka::exec<Acc>(queue, workDiv, PrefixSumBuffKernel{}, sizes_sum_view, data_prefix_sum_buff);
    ::alpaka::wait(queue);

    return prefix_sum_buff;
}

}  // namespace traccc::alpaka
