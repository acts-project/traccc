/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/track_candidate_container.hpp"
#include "traccc/fitting/device/fit.hpp"
#include "traccc/fitting/status_codes.hpp"

namespace traccc::device {

TRACCC_HOST_DEVICE inline void fit_prelude(
    const global_index_t globalIndex,
    vecmem::data::vector_view<const unsigned int> param_ids_view,
    edm::track_candidate_container<default_algebra>::const_view
        track_candidates_view,
    track_state_container_types::view track_states_view,
    vecmem::data::vector_view<unsigned int> param_liveness_view) {

    track_state_container_types::device track_states(track_states_view);

    if (globalIndex >= track_states.size()) {
        return;
    }

    vecmem::device_vector<const unsigned int> param_ids(param_ids_view);
    vecmem::device_vector<unsigned int> param_liveness(param_liveness_view);

    const unsigned int param_id = param_ids.at(globalIndex);

    auto track_states_per_track = track_states.at(param_id).items;

    const edm::track_candidate_collection<default_algebra>::const_device
        track_candidates{track_candidates_view.tracks};
    const measurement_collection_types::const_device measurements{
        track_candidates_view.measurements};
    for (unsigned int meas_idx :
         track_candidates.measurement_indices().at(param_id)) {
        track_states_per_track.emplace_back(measurements.at(meas_idx));
    }

    // TODO: Set other stuff in the header?
    track_states.at(param_id).header.fit_params =
        track_candidates.at(param_id).params();
    param_liveness.at(param_id) = 1u;
}

}  // namespace traccc::device
