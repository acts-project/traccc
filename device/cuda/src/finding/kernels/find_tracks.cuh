/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/finding/device/find_tracks.hpp"
#include "traccc/finding/finding_config.hpp"

namespace traccc::cuda::kernels {

template <typename detector_t>
__global__ void find_tracks(const finding_config cfg,
                            device::find_tracks_payload<detector_t> payload);
}  // namespace traccc::cuda::kernels
