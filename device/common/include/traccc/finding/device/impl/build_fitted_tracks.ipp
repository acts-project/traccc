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

TRACCC_HOST_DEVICE inline void build_fitted_tracks(
    const global_index_t globalIndex,
    const build_fitted_tracks_payload& payload) {

    const measurement_collection_types::const_device measurements(
        payload.measurements_view);

    const bound_track_parameters_collection_types::const_device track_params(
        payload.track_param_view);
    const bound_track_parameters_collection_types::const_device seeds(
        payload.seeds_view);

    const vecmem::device_vector<const candidate_link> links(payload.links_view);

    const vecmem::device_vector<const unsigned int> tips(payload.tips_view);

    track_state_container_types::device track_states(payload.track_states_view);

    if (globalIndex >= tips.size()) {
        return;
    }

    const auto tip = tips.at(globalIndex);

    auto track = track_states.at(globalIndex).items;
    auto header = track_states.at(globalIndex).header;

    // Get the link corresponding to tip
    unsigned int link_idx = tip;
    auto L = links.at(link_idx);
    const unsigned int n_meas = measurements.size();

    // Track summary variables
    scalar ndf_sum = 0.f;
    scalar chi2_sum = 0.f;

    // Reversely iterate to fill the track candidates
    for (auto it = track.rbegin(); it != track.rend(); it++) {
        if (L.meas_idx >= n_meas) {
            it->is_hole = true;
        } else {
            *it = track_state<default_algebra>(measurements.at(L.meas_idx));
            it->is_hole = false;
            it->filtered_chi2() = L.chi2;
            it->filtered() = track_params.at(link_idx);

            // Sanity check on chi2
            assert(L.chi2 < std::numeric_limits<traccc::scalar>::max());
            assert(L.chi2 >= 0.f);

            ndf_sum +=
                static_cast<scalar>(measurements.at(L.meas_idx).meas_dim);
            chi2_sum += L.chi2;
        }

        // Break the loop if the iterator is at the first candidate and fill the
        // seed and track quality
        if (it != track.rend() - 1) {
            link_idx = L.previous_candidate_idx;
            L = links.at(link_idx);
        }
    }

    header.fit_outcome = fitter_outcome::SUCCESS;
    header.fit_params = seeds.at(L.seed_idx);
    header.trk_quality.ndf = ndf_sum - 5.f;
    header.trk_quality.chi2 = chi2_sum;
    header.trk_quality.pval =
        prob(header.trk_quality.chi2, header.trk_quality.ndf);
    header.trk_quality.n_holes = L.n_skipped;
}

}  // namespace traccc::device
