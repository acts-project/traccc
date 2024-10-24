/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/device/prune_tracks.hpp"

namespace traccc::cuda::kernels {

__global__ void prune_tracks(device::prune_tracks_payload payload);
}
