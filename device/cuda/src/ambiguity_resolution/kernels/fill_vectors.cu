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

    const edm::track_candidate_collection<default_algebra>::const_device
        track_candidates(payload.track_candidates_view.tracks);

    const auto globalIndex = details::global_index1();
    if (globalIndex >= track_candidates.size()) {
        return;
    }

    const measurement_collection_types::const_device measurements(
        payload.track_candidates_view.measurements);
    const auto track = track_candidates.at(globalIndex);

    vecmem::jagged_device_vector<std::size_t> meas_ids(payload.meas_ids_view);
    vecmem::device_vector<std::size_t> flat_meas_ids(
        payload.flat_meas_ids_view);
    vecmem::device_vector<traccc::scalar> pvals(payload.pvals_view);
    vecmem::device_vector<std::size_t> n_meas(payload.n_meas_view);
    vecmem::device_vector<int> status(payload.status_view);

    pvals.at(globalIndex) = track.pval();

    if (track.measurement_indices().size() < cfg.min_meas_per_track) {
        status.at(globalIndex) = 0;
    } else {
        for (const unsigned int meas_idx :
             track_candidates.measurement_indices().at(globalIndex)) {
            meas_ids.at(globalIndex)
                .push_back(measurements.at(meas_idx).measurement_id);
            flat_meas_ids.push_back(measurements.at(meas_idx).measurement_id);
        }
        n_meas.at(globalIndex) = track.measurement_indices().size();
    }
}
}  // namespace traccc::cuda::kernels
