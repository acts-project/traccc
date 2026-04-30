/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "../../../utils/global_index.hpp"
#include "../kalman_track_follower.hpp"

// Project include(s).
#include "traccc/finding/device/kalman_track_follower.hpp"

namespace traccc::cuda {
namespace kernels {

template <typename propagator_t>
__global__ __launch_bounds__(128) void kalman_track_follower(
    const finding_config cfg,
    device::kalman_track_follower_payload<propagator_t> payload) {

    device::kalman_track_follower<propagator_t>(details::global_index1(), cfg,
                                                payload);
}

}  // namespace kernels

template <typename propagator_t>
void kalman_track_follower(
    const dim3& grid_size, const dim3& block_size, std::size_t shared_mem_size,
    const cudaStream_t& stream, const finding_config cfg,
    device::kalman_track_follower_payload<propagator_t> payload) {

    kernels::kalman_track_follower<<<grid_size, block_size, shared_mem_size,
                                     stream>>>(cfg, payload);
}
}  // namespace traccc::cuda
