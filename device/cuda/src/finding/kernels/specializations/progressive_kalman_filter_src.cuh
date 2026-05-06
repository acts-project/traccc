/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "../../../utils/global_index.hpp"
#include "../progressive_kalman_filter.hpp"

// Project include(s).
#include "traccc/finding/device/progressive_kalman_filter.hpp"

namespace traccc::cuda {
namespace kernels {

template <typename propagator_t>
__global__ __launch_bounds__(128) void progressive_kalman_filter(
    const finding_config cfg,
    device::progressive_kalman_filter_payload<propagator_t> payload) {

    device::progressive_kalman_filter<propagator_t>(details::global_index1(),
                                                    cfg, payload);
}

}  // namespace kernels

template <typename propagator_t>
void progressive_kalman_filter(
    const dim3& grid_size, const dim3& block_size, std::size_t shared_mem_size,
    const cudaStream_t& stream, const finding_config cfg,
    device::progressive_kalman_filter_payload<propagator_t> payload) {

    kernels::progressive_kalman_filter<<<grid_size, block_size, shared_mem_size,
                                         stream>>>(cfg, payload);
}
}  // namespace traccc::cuda
