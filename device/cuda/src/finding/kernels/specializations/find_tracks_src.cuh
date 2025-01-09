/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/finding/device/find_tracks.hpp"

// Local include(s).
#include "../../../utils/barrier.hpp"
#include "../../../utils/thread_id.hpp"

// System include(s).
#include <utility>

namespace traccc::cuda::kernels {

template <typename detector_t>
__global__ void find_tracks(const finding_config cfg,
                            device::find_tracks_payload<detector_t> payload) {
    __shared__ unsigned int shared_candidates_size;
    extern __shared__ unsigned int s[];
    unsigned int* shared_num_candidates = s;
    std::pair<unsigned int, unsigned int>* shared_candidates =
        reinterpret_cast<std::pair<unsigned int, unsigned int>*>(
            &shared_num_candidates[blockDim.x]);

    cuda::barrier barrier;
    details::thread_id1 thread_id;

    device::find_tracks<detector_t>(
        thread_id, barrier, cfg, payload,
        {shared_num_candidates, shared_candidates, shared_candidates_size});
}
}  // namespace traccc::cuda::kernels
