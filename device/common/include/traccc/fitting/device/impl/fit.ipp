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
TRACCC_HOST_DEVICE inline void fit(
    const global_index_t globalIndex,
    typename fitter_t::detector_type::view_type det_data,
    const typename fitter_t::bfield_type field_data,
    const typename fitter_t::config_type cfg,
    track_candidate_container_types::const_view track_candidates_view,
    const vecmem::data::vector_view<const unsigned int>& param_ids_view,
    track_state_container_types::view track_states_view) {

    typename fitter_t::detector_type det(det_data);

    track_candidate_container_types::const_device track_candidates(
        track_candidates_view);

    vecmem::device_vector<const unsigned int> param_ids(param_ids_view);

    track_state_container_types::device track_states(track_states_view);

    fitter_t fitter(det, field_data, cfg);

    if (globalIndex >= track_states.size()) {
        return;
    }

    const unsigned int param_id = param_ids.at(globalIndex);

    // Track candidates per track
    const auto& track_candidates_per_track =
        track_candidates.at(param_id).items;

    // Seed parameter
    const auto& seed_param = track_candidates.at(param_id).header.seed_params;

    // Track states per track
    auto track_states_per_track = track_states.at(param_id).items;

    for (auto& cand : track_candidates_per_track) {
        track_states_per_track.emplace_back(cand);
    }

    typename fitter_t::state fitter_state(track_states_per_track);

    // Run fitting
    kalman_fitter_status fit_status = fitter.fit(seed_param, fitter_state);

    if (fitter_state.m_fit_res.fit_outcome == fitter_outcome::SUCCESS) {
        assert(fit_status == kalman_fitter_status::SUCCESS);
    }

    // Get the final fitting information
    track_states.at(param_id).header = fitter_state.m_fit_res;
}

}  // namespace traccc::device
