/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "../../../utils/global_index.hpp"
#include "../condense_tracks.cuh"

// Project include(s).
#include "traccc/finding/device/condense_tracks.hpp"

namespace traccc::cuda {
namespace kernels {

template <typename detector_t>
__global__ void condense_tracks(
    const __grid_constant__ device::condense_tracks_payload<detector_t>
        payload) {

    device::condense_tracks<detector_t>(details::global_index1(), payload);
}

}  // namespace kernels

template <typename detector_t>
void condense_tracks(
    const dim3& grid_size, const dim3& block_size, std::size_t shared_mem_size,
    const cudaStream_t& stream,
    const device::condense_tracks_payload<detector_t> payload) {
    kernels::
        condense_tracks<<<grid_size, block_size, shared_mem_size, stream>>>(
            payload);
}
}  // namespace traccc::cuda
