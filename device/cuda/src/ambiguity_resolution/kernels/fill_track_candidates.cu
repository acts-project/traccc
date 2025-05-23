/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/global_index.hpp"
#include "fill_track_candidates.cuh"

namespace traccc::cuda::kernels {

__global__ void fill_track_candidates(
    device::fill_track_candidates_payload payload) {

    device::fill_track_candidates(details::global_index1(), payload);
}
}  // namespace traccc::cuda::kernels
