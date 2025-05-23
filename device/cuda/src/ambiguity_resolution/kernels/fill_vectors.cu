/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/global_index.hpp"
#include "fill_vectors.cuh"

// Project include(s).
#include "traccc/ambiguity_resolution/ambiguity_resolution_config.hpp"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>

namespace traccc::cuda::kernels {

__global__ void fill_vectors(const ambiguity_resolution_config cfg,
                             device::fill_vectors_payload payload) {

    track_candidate_container_types::const_device track_candidates(
        payload.track_candidates_view);

    auto globalIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if (globalIndex >= track_candidates.size()) {
        return;
    }

    vecmem::jagged_device_vector<std::size_t> meas_ids(payload.meas_ids_view);
    vecmem::device_vector<std::size_t> flat_meas_ids(
        payload.flat_meas_ids_view);
    vecmem::device_vector<traccc::scalar> pvals(payload.pvals_view);
    vecmem::device_vector<std::size_t> n_meas(payload.n_meas_view);
    vecmem::device_vector<int> status(payload.status_view);

    pvals.at(globalIndex) =
        track_candidates.at(globalIndex).header.trk_quality.pval;

    const auto& candidates = track_candidates.at(globalIndex).items;
    const unsigned int n_cands = candidates.size();

    if (n_cands < cfg.min_meas_per_track) {
        status.at(globalIndex) = 0;
    } else {
        for (const auto& cand : candidates) {
            meas_ids.at(globalIndex).push_back(cand.measurement_id);
            flat_meas_ids.push_back(cand.measurement_id);
        }
        n_meas.at(globalIndex) = n_cands;
    }
}
}  // namespace traccc::cuda::kernels
