/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/barrier.hpp"
#include "../../utils/global_index.hpp"
#include "sort_updated_tracks.cuh"

namespace traccc::cuda::kernels {

__global__ void sort_updated_tracks(
    device::sort_updated_tracks_payload payload) {

    cuda::barrier barrier;
    extern __shared__ unsigned int shared_mem_tracks[];

    device::sort_updated_tracks(details::global_index1(), barrier, payload,
                                shared_mem_tracks);
}

}  // namespace traccc::cuda::kernels
