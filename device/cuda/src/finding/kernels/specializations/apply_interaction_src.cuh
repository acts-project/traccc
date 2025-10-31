/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "../../../utils/global_index.hpp"
#include "../apply_interaction.hpp"

// Project include(s).
#include "traccc/finding/device/apply_interaction.hpp"

namespace traccc::cuda {
namespace kernels {

template <typename detector_t>
__global__ void apply_interaction(
    const __grid_constant__ finding_config cfg,
    device::apply_interaction_payload<detector_t> payload) {

    device::apply_interaction<detector_t>(details::global_index1(), cfg,
                                          payload);
}

}  // namespace kernels

template <typename detector_t>
void apply_interaction(const dim3& grid_size, const dim3& block_size,
                       std::size_t shared_mem_size, const cudaStream_t& stream,
                       const finding_config cfg,
                       device::apply_interaction_payload<detector_t> payload) {
    kernels::
        apply_interaction<<<grid_size, block_size, shared_mem_size, stream>>>(
            cfg, payload);
}
}  // namespace traccc::cuda
