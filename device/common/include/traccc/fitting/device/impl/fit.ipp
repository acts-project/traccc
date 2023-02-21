/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

template <typename fitter_t, typename detector_view_t>
TRACCC_HOST_DEVICE inline void fit(
    std::size_t globalIndex, detector_view_t det_data,
    vecmem::data::jagged_vector_view<typename fitter_t::intersection_type>
        nav_candidates_buffer,
    track_candidate_container_types::const_view track_candidates_view,
    track_state_container_types::view track_states_view) {

    typename fitter_t::detector_type det(det_data);

    vecmem::jagged_device_vector<typename fitter_t::intersection_type>
        nav_candidates(nav_candidates_buffer);

    track_candidate_container_types::const_device track_candidates(
        track_candidates_view);

    track_state_container_types::device track_states(track_states_view);

    fitter_t fitter(det);

    if (globalIndex >= track_states.size()) {
        return;
    }

    // Track candidates per track
    const auto& track_candidates_per_track =
        track_candidates[globalIndex].items;

    // Seed parameter
    const auto& seed_param = track_candidates[globalIndex].header;

    // Track states per track
    auto track_states_per_track = track_states[globalIndex].items;

    for (auto& cand : track_candidates_per_track) {
        track_states_per_track.emplace_back(cand);
    }

    typename fitter_t::state fitter_state(track_states_per_track);

    fitter.fit(seed_param, fitter_state, nav_candidates.at(globalIndex));
}

}  // namespace traccc::device