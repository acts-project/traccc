/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

TRACCC_HOST_DEVICE inline void fill_track_candidates(
    const global_index_t globalIndex,
    const fill_track_candidates_payload& payload) {

    track_candidate_container_types::const_device track_candidates(
        payload.track_candidates_view);

    if (globalIndex >= payload.n_accepted) {
        return;
    }

    vecmem::device_vector<const unsigned int> sorted_ids(
        payload.sorted_ids_view);
    track_candidate_container_types::device res_track_candidates(
        payload.res_track_candidates_view);

    // Set header
    const auto tid = sorted_ids[globalIndex];
    res_track_candidates.at(globalIndex).header =
        track_candidates.at(tid).header;

    // Set items
    auto res_cands = res_track_candidates.at(globalIndex).items;
    const auto& cands = track_candidates.at(tid).items;
    const auto n_cands = cands.size();
    res_cands.resize(n_cands);

    for (unsigned int i = 0; i < n_cands; i++) {
        res_cands.at(i) = cands.at(i);
    }
}

}  // namespace traccc::device
