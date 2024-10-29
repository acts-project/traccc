/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/utils.hpp"

// Project include(s).
#include "traccc/alpaka/utils/make_prefix_sum_buff.hpp"
#include "traccc/device/make_prefix_sum_buffer.hpp"

namespace traccc::alpaka {

struct PrefixSumBuffKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        vecmem::data::vector_view<const device::prefix_sum_size_t> sizes_view,
        vecmem::data::vector_view<device::prefix_sum_element_t> ps_view) const {
        auto const globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0u];
        device::fill_prefix_sum(globalThreadIdx, sizes_view, ps_view);
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
    copy.setup(prefix_sum_buff)->wait();
    auto data_prefix_sum_buff = vecmem::get_data(prefix_sum_buff);

    // Setup Alpaka
    auto const deviceProperties = ::alpaka::getAccDevProps<Acc>(
        ::alpaka::getDevByIdx(::alpaka::Platform<Acc>{}, 0u));
    auto const threadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];

    // Fixed number of threads per block.
    auto const blocksPerGrid =
        (totalSize + threadsPerBlock - 1) / threadsPerBlock;
    auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

    ::alpaka::exec<Acc>(queue, workDiv, PrefixSumBuffKernel{}, sizes_sum_view,
                        data_prefix_sum_buff);
    ::alpaka::wait(queue);

    return prefix_sum_buff;
}

}  // namespace traccc::alpaka
