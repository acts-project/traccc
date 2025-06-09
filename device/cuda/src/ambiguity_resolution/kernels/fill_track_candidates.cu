/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/global_index.hpp"
#include "fill_track_candidates.cuh"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

namespace traccc::cuda::kernels {

__global__ void fill_track_candidates(
    device::fill_track_candidates_payload payload) {

    const auto globalIndex = details::global_index1();
    if (globalIndex >= payload.n_accepted) {
        return;
    }

    // Set up the device objects.
    vecmem::device_vector<const unsigned int> sorted_ids(
        payload.sorted_ids_view);
    edm::track_candidate_collection<default_algebra>::const_device
        track_candidates(payload.track_candidates_view);
    edm::track_candidate_collection<default_algebra>::device
        res_track_candidates(payload.res_track_candidates_view);

    // Copy the appropriate track candidate.
    res_track_candidates.at(globalIndex) =
        track_candidates.at(sorted_ids.at(globalIndex));
}

}  // namespace traccc::cuda::kernels
