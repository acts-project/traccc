/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/kokkos/utils/definitions.hpp"

// Project include(s).
#include "traccc/kokkos/utils/make_prefix_sum_buff.hpp"
#include "traccc/device/make_prefix_sum_buffer.hpp"

namespace traccc::kokkos {

namespace kernels {

/// CUDA kernel for running @c traccc::device::fill_prefix_sum
/*
__global__ void fill_prefix_sum(
    vecmem::data::vector_view<const device::prefix_sum_size_t> sizes_view,
    vecmem::data::vector_view<device::prefix_sum_element_t> ps_view) {

    device::fill_prefix_sum(threadIdx.x + blockIdx.x * blockDim.x, sizes_view,
                            ps_view);
}

*/
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
    //kernels::fill_prefix_sum<<<(sizes_sum_view.size() / 32) + 1, 32>>>(
    //    sizes_sum_view, prefix_sum_buff);
    uint64_t num_blocks = (sizes_sum_view.size() / 32) + 1;
    uint64_t num_threads = 32;
    auto data_prefix_sum_buff = vecmem::get_data(prefix_sum_buff);
    Kokkos::parallel_for("fill_prefix_sum", team_policy(num_blocks, num_threads),
      KOKKOS_LAMBDA (const member_type &team_member) {
        device::fill_prefix_sum(team_member.league_rank() * team_member.team_size() + team_member.team_rank(),
                                sizes_sum_view, data_prefix_sum_buff);
      }  
    );
    return prefix_sum_buff;
}

}  // namespace traccc::kokkos
