/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/cuda/utils/definitions.hpp"
#include "utils.hpp"

// Project include(s).
#include "traccc/cuda/utils/make_prefix_sum_buff.hpp"
#include "traccc/device/make_prefix_sum_buffer.hpp"

namespace traccc::cuda {

namespace kernels {

/// CUDA kernel for running @c traccc::device::fill_prefix_sum
__global__ void fill_prefix_sum(
    vecmem::data::vector_view<const device::prefix_sum_size_t> sizes_view,
    vecmem::data::vector_view<device::prefix_sum_element_t> ps_view) {

    device::fill_prefix_sum(threadIdx.x + blockIdx.x * blockDim.x, sizes_view,
                            ps_view);
}

}  // namespace kernels

vecmem::data::vector_buffer<device::prefix_sum_element_t> make_prefix_sum_buff(
    const std::vector<device::prefix_sum_size_t>& sizes, vecmem::copy& copy,
    const traccc::memory_resource& mr) {

    const device::prefix_sum_buffer_t make_sum_result =
        device::make_prefix_sum_buffer(sizes, copy, mr);
    const vecmem::data::vector_view<const device::prefix_sum_size_t>
        sizes_sum_view = make_sum_result.view;
    const unsigned int totalSize = make_sum_result.totalSize;

    // Create buffer and view objects for prefix sum vector
    vecmem::data::vector_buffer<device::prefix_sum_element_t> prefix_sum_buff(
        totalSize, mr.main);
    copy.setup(prefix_sum_buff);

    // Fill the prefix sum vector
    static const unsigned int threadsPerBlock = 32;
    const unsigned int blocks =
        (sizes_sum_view.size() + threadsPerBlock - 1) / threadsPerBlock;
    kernels::fill_prefix_sum<<<blocks, threadsPerBlock>>>(sizes_sum_view,
                                                          prefix_sum_buff);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    return prefix_sum_buff;
}

vecmem::data::vector_buffer<device::prefix_sum_element_t> make_prefix_sum_buff(
    const std::vector<device::prefix_sum_size_t>& sizes, vecmem::copy& copy,
    const traccc::memory_resource& mr, const stream& str) {

    const device::prefix_sum_buffer_t make_sum_result =
        device::make_prefix_sum_buffer(sizes, copy, mr);
    const vecmem::data::vector_view<const device::prefix_sum_size_t>
        sizes_sum_view = make_sum_result.view;
    const unsigned int totalSize = make_sum_result.totalSize;

    // Create buffer and view objects for prefix sum vector
    vecmem::data::vector_buffer<device::prefix_sum_element_t> prefix_sum_buff(
        totalSize, mr.main);
    copy.setup(prefix_sum_buff);

    // Fill the prefix sum vector
    static const unsigned int threadsPerBlock = 32;
    const unsigned int blocks =
        (sizes_sum_view.size() + threadsPerBlock - 1) / threadsPerBlock;
    kernels::fill_prefix_sum<<<blocks, threadsPerBlock, 0,
                               details::get_stream(str)>>>(sizes_sum_view,
                                                           prefix_sum_buff);
    CUDA_ERROR_CHECK(cudaGetLastError());

    return prefix_sum_buff;
}

}  // namespace traccc::cuda
