/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/candidate_link.hpp"

namespace traccc::device {

template <typename config_t>
TRACCC_DEVICE inline void build_tracks(std::size_t globalIndex,
                                       const config_t cfg,
                                       const build_tracks_payload& payload) {

    measurement_collection_types::const_device measurements(
        payload.measurements_view);

    bound_track_parameters_collection_types::const_device seeds(
        payload.seeds_view);

    vecmem::jagged_device_vector<const candidate_link> links(
        payload.links_view);

    vecmem::device_vector<const typename candidate_link::link_index_type> tips(
        payload.tips_view);

    track_candidate_container_types::device track_candidates(
        payload.track_candidates_view);

    vecmem::device_vector<unsigned int> valid_indices(
        payload.valid_indices_view);

    if (globalIndex >= tips.size()) {
        return;
    }

    const auto tip = tips.at(globalIndex);
    auto& seed = track_candidates[globalIndex].header;
    auto cands_per_track = track_candidates[globalIndex].items;

    // Get the link corresponding to tip
    auto L = links[tip.first][tip.second];
    const unsigned int n_meas = measurements.size();

    // Count the number of skipped steps
    unsigned int n_skipped{0u};
    while (true) {
        if (L.meas_idx > n_meas) {
            n_skipped++;
        }

        if (L.previous.first == 0u) {
            break;
        }

        L = links[L.previous.first][L.previous.second];
    }

    // Retrieve tip
    L = links[tip.first][tip.second];

    const unsigned int n_cands = tip.first + 1 - n_skipped;

    // Resize the candidates with the exact size
    cands_per_track.resize(n_cands);

    unsigned int i = 0;

    // Reversely iterate to fill the track candidates
    for (auto it = cands_per_track.rbegin(); it != cands_per_track.rend();
         it++) {
        i++;

        while (L.meas_idx > n_meas &&
               L.previous.first !=
                   std::numeric_limits<
                       candidate_link::link_index_type::first_type>::max()) {

            L = links[L.previous.first][L.previous.second];
        }

        // Break if the measurement is still invalid
        if (L.meas_idx > measurements.size()) {
            break;
        }

        auto& cand = *it;
        cand = {measurements.at(L.meas_idx)};

        // Break the loop if the iterator is at the first candidate and fill the
        // seed
        if (it == cands_per_track.rend() - 1) {
            seed = seeds.at(L.previous.second);
            break;
        }

        L = links[L.previous.first][L.previous.second];
    }

    // Criteria for valid tracks
    if (n_cands >= cfg.min_track_candidates_per_track &&
        n_cands <= cfg.max_track_candidates_per_track) {

        vecmem::device_atomic_ref<unsigned int> num_valid_tracks(
            *payload.n_valid_tracks);

        const unsigned int pos = num_valid_tracks.fetch_add(1);
        valid_indices[pos] = globalIndex;
    }
}

}  // namespace traccc::device
