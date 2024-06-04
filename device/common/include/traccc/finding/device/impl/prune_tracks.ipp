/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

TRACCC_DEVICE inline void prune_tracks(
    std::size_t globalIndex,
    track_candidate_container_types::const_view track_candidates_view,
    vecmem::data::vector_view<const unsigned int> valid_indices_view,
    track_candidate_container_types::view prune_candidates_view) {

    track_candidate_container_types::const_device track_candidates(
        track_candidates_view);
    vecmem::device_vector<const unsigned int> valid_indices(valid_indices_view);
    track_candidate_container_types::device prune_candidates(
        prune_candidates_view);

    if (globalIndex >= prune_candidates.size()) {
        return;
    }

    const auto idx = valid_indices.at(globalIndex);

    auto& seed = track_candidates.at(idx).header;
    auto cands_per_track = track_candidates.at(idx).items;

    // Copy candidates
    prune_candidates[globalIndex].header = seed;
    const unsigned int n_cands = cands_per_track.size();
    prune_candidates[globalIndex].items.resize(n_cands);

    for (unsigned int i = 0; i < n_cands; i++) {
        prune_candidates.at(globalIndex).items.at(i) = cands_per_track.at(i);
    }
}

}  // namespace traccc::device