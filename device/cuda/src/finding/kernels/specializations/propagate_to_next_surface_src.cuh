/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "../../../utils/global_index.hpp"
#include "../propagate_to_next_surface.cuh"

// Project include(s).
#include "traccc/finding/device/propagate_to_next_surface.hpp"

namespace traccc::cuda::kernels {

template <typename propagator_t, typename bfield_t>
__global__ void propagate_to_next_surface(
    const finding_config cfg,
    device::propagate_to_next_surface_payload<propagator_t, bfield_t> payload) {

    device::propagate_to_next_surface<propagator_t, bfield_t>(
        details::global_index1(), cfg, payload);
}

}  // namespace traccc::cuda::kernels
