/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "build_tracks.cuh"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/finding/device/build_tracks.hpp"
#include "traccc/finding/finding_config.hpp"

namespace traccc::cuda::kernels {

__global__ void build_tracks(const finding_config cfg,
                             device::build_tracks_payload payload) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::build_tracks(gid, cfg, payload);
}
}  // namespace traccc::cuda::kernels
