/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/utils/prob.hpp"

namespace traccc::device {

TRACCC_HOST_DEVICE inline void build_tracks(
    const global_index_t globalIndex, const build_tracks_payload& payload) {

    const measurement_collection_types::const_device measurements(
        payload.tracks_view.measurements);

    const bound_track_parameters_collection_types::const_device seeds(
        payload.seeds_view);

    const vecmem::device_vector<const candidate_link> links(payload.links_view);

    const vecmem::device_vector<const unsigned int> tips(payload.tips_view);

    edm::track_collection<default_algebra>::device track_candidates(
        payload.tracks_view.tracks);

    if (globalIndex >= tips.size()) {
        return;
    }

    const auto tip = tips.at(globalIndex);
    auto track = track_candidates.at(globalIndex);

    // Get the link corresponding to tip
    auto L = links.at(tip);
    const unsigned int n_meas = measurements.size();

    // Track summary variables
    scalar ndf_sum = 0.f;
    scalar chi2_sum = 0.f;

    // Reversely iterate to fill the track candidates
    for (auto it = track.constituent_links().rbegin();
         it != track.constituent_links().rend(); it++) {

        while (L.meas_idx >= n_meas && L.step != 0u) {

            L = links.at(L.previous_candidate_idx);
        }

        assert(L.meas_idx < n_meas);

        *it = {edm::track_constituent_link::measurement, L.meas_idx};

        // Sanity check on chi2
        assert(L.chi2 < std::numeric_limits<traccc::scalar>::max());
        assert(L.chi2 >= 0.f);

        ndf_sum += static_cast<scalar>(measurements.at(it->index).meas_dim);
        chi2_sum += L.chi2;

        // Break the loop if the iterator is at the first candidate and fill the
        // seed and track quality
        if (it == track.constituent_links().rend() - 1) {
            track.fit_outcome() = track_fit_outcome::UNKNOWN;
            track.params() = seeds.at(L.seed_idx);
            track.ndf() = ndf_sum - 5.f;
            track.chi2() = chi2_sum;
            track.pval() = prob(track.chi2(), track.ndf());
            track.nholes() = L.n_skipped;
        } else {
            L = links.at(L.previous_candidate_idx);
        }
    }

#ifndef NDEBUG
    // Assert that we did not make any duplicate track states.
    for (const auto& i : track.constituent_links()) {
        assert(i.type == edm::track_constituent_link::measurement);
        for (const auto& j : track.constituent_links()) {
            assert(j.type == edm::track_constituent_link::measurement);
            if (i.index != j.index) {
                // TODO: Re-enable me!
                // assert(measurements.at(i->index).measurement_id !=
                //       measurement.at(j->index).measurement_id);
            }
        }
    }
#endif
}

}  // namespace traccc::device
