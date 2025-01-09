/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/finding/device/prune_tracks.hpp"

namespace traccc::cuda::kernels {

__global__ void prune_tracks(device::prune_tracks_payload payload);

}
