/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/fitting/status_codes.hpp"

namespace traccc::device {

template <typename fitter_t>
TRACCC_HOST_DEVICE inline void fit(const global_index_t globalIndex,
                                   const typename fitter_t::config_type cfg,
                                   const fit_payload<fitter_t>& payload) {

    typename fitter_t::detector_type det(payload.det_data);

    const edm::track_candidate_collection<default_algebra>::const_device
        track_candidates(payload.track_candidates_view);
    const measurement_collection_types::const_device measurements{
        payload.measurements_view};

    vecmem::device_vector<const unsigned int> param_ids(payload.param_ids_view);

    track_state_container_types::device track_states(payload.track_states_view);

    fitter_t fitter(det, payload.field_data, cfg);

    if (globalIndex >= track_states.size()) {
        return;
    }

    const unsigned int param_id = param_ids.at(globalIndex);
    const edm::track_candidate_collection<
        default_algebra>::const_device::const_proxy_type track =
        track_candidates.at(param_id);

    // Track states per track
    auto track_states_per_track = track_states.at(param_id).items;

    for (unsigned int midx : track.measurement_indices()) {
        track_states_per_track.emplace_back(measurements.at(midx));
    }

    typename fitter_t::state fitter_state(
        track_states_per_track, *(payload.barcodes_view.ptr() + param_id));

    // Run fitting
    [[maybe_unused]] kalman_fitter_status fit_status =
        fitter.fit(track.params(), fitter_state);

    if (fitter_state.m_fit_res.fit_outcome == fitter_outcome::SUCCESS) {
        assert(fit_status == kalman_fitter_status::SUCCESS);
    }

    // Get the final fitting information
    track_states.at(param_id).header = fitter_state.m_fit_res;
}

}  // namespace traccc::device
