/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/barrier.hpp"
#include "../../utils/global_index.hpp"
#include "rearrange_tracks.cuh"

namespace traccc::cuda::kernels {

__global__ void rearrange_tracks(device::rearrange_tracks_payload payload) {

    cuda::barrier barrier;

    device::rearrange_tracks(details::global_index1(), barrier, payload);
}

}  // namespace traccc::cuda::kernels
