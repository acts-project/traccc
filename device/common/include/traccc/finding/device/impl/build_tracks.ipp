/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

TRACCC_DEVICE inline void build_tracks(
    std::size_t globalIndex,
    measurement_container_types::const_view measurements_view,
    bound_track_parameters_collection_types::const_view seeds_view,
    vecmem::data::jagged_vector_view<const candidate_link> links_view,
    vecmem::data::jagged_vector_view<const unsigned int> param_to_link_view,
    vecmem::data::vector_view<const typename candidate_link::link_index_type>
        tips_view,
    track_candidate_container_types::view track_candidates_view) {

    measurement_container_types::const_device measurements(measurements_view);

    bound_track_parameters_collection_types::const_device seeds(seeds_view);

    vecmem::jagged_device_vector<const candidate_link> links(links_view);

    vecmem::jagged_device_vector<const unsigned int> param_to_link(
        param_to_link_view);

    vecmem::device_vector<const typename candidate_link::link_index_type> tips(
        tips_view);

    track_candidate_container_types::device track_candidates(
        track_candidates_view);

    if (globalIndex >= tips.size()) {
        return;
    }

    const auto tip = tips.at(globalIndex);
    auto& seed = track_candidates[globalIndex].header;
    auto cands_per_track = track_candidates[globalIndex].items;

    // Get the link corresponding to tip
    auto L = links[tip.first][tip.second];

    // Resize the candidates with the exact size
    cands_per_track.resize(tip.first + 1);

    // Reversely iterate to fill the track candidates
    for (auto it = cands_per_track.rbegin(); it != cands_per_track.rend();
         it++) {

        auto& cand = *it;
        cand = {L.surface_link, measurements.at(L.meas_link)};

        // Break the loop if the iterator is at the first candidate and fill the
        // seed
        if (it == cands_per_track.rend() - 1) {
            seed = seeds.at(L.previous.second);
            break;
        }

        const auto l_pos = param_to_link[L.previous.first][L.previous.second];

        L = links[L.previous.first][l_pos];
    }
}

}  // namespace traccc::device