/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/kokkos/utils/definitions.hpp"

// Project include(s).
#include "traccc/device/make_prefix_sum_buffer.hpp"
#include "traccc/kokkos/utils/make_prefix_sum_buff.hpp"

namespace traccc::kokkos {

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
    // kernels::fill_prefix_sum<<<(sizes_sum_view.size() / 32) + 1, 32>>>(
    //    sizes_sum_view, prefix_sum_buff);
    uint64_t num_blocks = (sizes_sum_view.size() / 32) + 1;
    uint64_t num_threads = 32;
    auto data_prefix_sum_buff = vecmem::get_data(prefix_sum_buff);
    Kokkos::parallel_for(
        "fill_prefix_sum", team_policy(num_blocks, num_threads),
        KOKKOS_LAMBDA(const member_type& team_member) {
            device::fill_prefix_sum(
                team_member.league_rank() * team_member.team_size() +
                    team_member.team_rank(),
                sizes_sum_view, data_prefix_sum_buff);
        });
    return prefix_sum_buff;
}

}  // namespace traccc::kokkos
