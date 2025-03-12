/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <vecmem/containers/device_vector.hpp>
namespace traccc::device {

TRACCC_DEVICE inline void count_tracks(const global_index_t globalIndex,
                                       const finding_config& cfg,
                                       const count_tracks_payload& payload) {

    const measurement_collection_types::const_device measurements(
        payload.measurements_view);

    const vecmem::jagged_device_vector<const candidate_link> links(
        payload.links_view);

    const vecmem::device_vector<const typename candidate_link::link_index_type>
        tips(payload.tips_view);

    vecmem::device_vector<unsigned int> valid_tip_idx(
        payload.valid_tip_idx_view);
    vecmem::device_vector<unsigned int> valid_tip_length(
        payload.valid_tip_length_view);

    if (globalIndex >= tips.size()) {
        return;
    }

    const auto tip = tips.at(globalIndex);

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

    const unsigned int num_meas = tip.first + 1 - n_skipped;

    bool success = true;

    // Reversely iterate to fill the track candidates
    for (std::size_t i = num_meas - 1; i < num_meas; --i) {
        while (L.meas_idx >= n_meas &&
               L.previous.first !=
                   std::numeric_limits<
                       candidate_link::link_index_type::first_type>::max()) {

            L = links[L.previous.first][L.previous.second];
        }

        // Break if the measurement is still invalid
        if (L.meas_idx >= measurements.size()) {
            success = false;
            break;
        }

        // Sanity check on chi2
        assert(L.chi2 < std::numeric_limits<traccc::scalar>::max());
        assert(L.chi2 >= 0.f);

        // Break the loop if the iterator is at the first candidate and fill the
        // seed and track quality
        if (i != 0) {
            L = links[L.previous.first][L.previous.second];
        }
    }

    // Criteria for valid tracks
    if (num_meas >= cfg.min_track_candidates_per_track &&
        num_meas <= cfg.max_track_candidates_per_track && success) {

        vecmem::device_atomic_ref<unsigned int> num_valid_tips(
            *payload.num_valid_tips);
        const unsigned int trk_pos = num_valid_tips.fetch_add(1);
        valid_tip_idx.at(trk_pos) = globalIndex;
        valid_tip_length.at(trk_pos) = num_meas;
    }
}

}  // namespace traccc::device
