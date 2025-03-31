/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <vecmem/containers/device_vector.hpp>
namespace traccc::device {

TRACCC_DEVICE inline void build_tracks(const global_index_t globalIndex,
                                       const finding_config& cfg,
                                       const build_tracks_payload& payload) {

    const measurement_collection_types::const_device measurements(
        payload.measurements_view);

    const bound_track_parameters_collection_types::const_device seeds(
        payload.seeds_view);

    const vecmem::device_vector<const candidate_link> links(payload.links_view);

    const vecmem::device_vector<const unsigned int> tips(payload.tips_view);

    track_candidate_container_types::device final_candidates(
        payload.final_candidates_view);

    if (globalIndex >= tips.size()) {
        return;
    }

    const auto tip = tips.at(globalIndex);

    const auto& L_origin = links.at(tip);

    // Get the link corresponding to tip
    auto L = L_origin;
    const unsigned int num_meas = final_candidates.at(globalIndex).items.size();
    const unsigned int n_meas = measurements.size();

    assert(num_meas >= cfg.min_track_candidates_per_track &&
           num_meas <= cfg.max_track_candidates_per_track);

    // Track summary variables
    scalar ndf_sum = 0.f;
    scalar chi2_sum = 0.f;

    // Reversely iterate to fill the track candidates
    for (unsigned int i = num_meas - 1; i < num_meas; --i) {
        while (L.meas_idx >= n_meas && L.step != 0u) {

            L = links.at(L.previous_candidate_idx);
        }

        assert(L.meas_idx < measurements.size());

        final_candidates.at(globalIndex).items.at(i) =
            measurements.at(L.meas_idx);

        // Sanity check on chi2
        assert(L.chi2 < std::numeric_limits<traccc::scalar>::max());
        assert(L.chi2 >= 0.f);

        ndf_sum += static_cast<scalar>(measurements.at(L.meas_idx).meas_dim);
        chi2_sum += L.chi2;

        if (i != 0) {
            L = links.at(L.previous_candidate_idx);
        }
    }

    final_candidates.at(globalIndex).header.seed_params =
        seeds.at(L_origin.seed_idx);
    final_candidates.at(globalIndex).header.trk_quality = {
        .ndf = ndf_sum - 5.f, .chi2 = chi2_sum, .n_holes = L_origin.n_skipped};

#ifndef NDEBUG
    // Assert that we did not make any duplicate track states.
    for (unsigned int i = 0; i < num_meas; ++i) {
        for (unsigned int j = 0; j < num_meas; ++j) {
            if (i != j) {
                // TODO: Re-enable me!
                // assert(cands_per_track.at(i).measurement_id !=
                //       cands_per_track.at(j).measurement_id);
            }
        }
    }
#endif
}

}  // namespace traccc::device
