/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "../../utils/barrier.hpp"
#include "traccc/cuda/utils/thread_id.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/device/find_tracks.hpp"
#include "traccc/geometry/detector.hpp"

namespace traccc::cuda::kernels {

template <typename detector_t>
__global__ void find_tracks(const finding_config cfg,
                            device::find_tracks_payload<detector_t> payload);
}  // namespace traccc::cuda::kernels
