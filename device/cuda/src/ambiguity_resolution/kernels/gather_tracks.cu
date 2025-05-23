/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/global_index.hpp"
#include "gather_tracks.cuh"

namespace traccc::cuda::kernels {

__global__ void gather_tracks(device::gather_tracks_payload payload) {

    device::gather_tracks(details::global_index1(), payload);
}

}  // namespace traccc::cuda::kernels
