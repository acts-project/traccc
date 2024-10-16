/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/device/apply_interaction.hpp"
#include "traccc/geometry/detector.hpp"

namespace traccc::cuda::kernels {

template <typename detector_t>
__global__ void apply_interaction(
    const finding_config cfg,
    device::apply_interaction_payload<detector_t> payload);
}  // namespace traccc::cuda::kernels
