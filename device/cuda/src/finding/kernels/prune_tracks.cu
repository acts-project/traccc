/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "prune_tracks.cuh"
#include "traccc/finding/device/prune_tracks.hpp"

namespace traccc::cuda::kernels {

__global__ void prune_tracks(device::prune_tracks_payload payload) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::prune_tracks(gid, payload);
}
}  // namespace traccc::cuda::kernels
