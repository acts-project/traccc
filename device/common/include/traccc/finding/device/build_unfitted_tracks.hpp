/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/global_index.hpp"

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/track_candidate_container.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/utils/prob.hpp"

// VecMem include(s).
#include <vecmem/containers/data/jagged_vector_view.hpp>
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Event Data) Payload for the @c traccc::device::build_tracks function
struct build_unfitted_tracks_payload {
    /**
     * @brief View object to the vector of measurements
     */
    bound_track_parameters_collection_types::const_view seeds_view;

    /**
     * @brief View object to the vector of candidate links
     */
    vecmem::data::vector_view<const candidate_link> links_view;

    /**
     * @brief View object to the vector of tips
     */
    vecmem::data::vector_view<const unsigned int> tips_view;

    /**
     * @brief View object to the vector of track candidates
     */
    edm::track_candidate_container<default_algebra>::view track_candidates_view;
};

/// Function for building full tracks from the link container:
/// The full tracks are built using the link container and tip link container.
/// Since every link holds an information of the link from the previous step,
/// we can build a full track by iterating from a tip link backwardly.
///
/// @param[in] globalIndex         The index of the current thread
/// @param[in] cfg                    Track finding config object
/// @param[inout] payload      The function call payload
///
TRACCC_HOST_DEVICE inline void build_unfitted_tracks(
    global_index_t globalIndex, const build_unfitted_tracks_payload& payload) {
    const measurement_collection_types::const_device measurements(
        payload.track_candidates_view.measurements);

    const bound_track_parameters_collection_types::const_device seeds(
        payload.seeds_view);

    const vecmem::device_vector<const candidate_link> links(payload.links_view);

    const vecmem::device_vector<const unsigned int> tips(payload.tips_view);

    edm::track_candidate_collection<default_algebra>::device track_candidates(
        payload.track_candidates_view.tracks);

    if (globalIndex >= tips.size()) {
        return;
    }

    const auto tip = tips.at(globalIndex);
    edm::track_candidate_collection<default_algebra>::device::proxy_type track =
        track_candidates.at(globalIndex);

    // Get the link corresponding to tip
    auto L = links.at(tip);
    const unsigned int n_meas = measurements.size();

    // Track summary variables
    scalar ndf_sum = 0.f;
    scalar chi2_sum = 0.f;

    // Reversely iterate to fill the track candidates
    for (auto it = track.measurement_indices().rbegin();
         it != track.measurement_indices().rend(); it++) {

        while (L.meas_idx >= n_meas && L.step != 0u) {

            L = links.at(L.previous_candidate_idx);
        }

        assert(L.meas_idx < n_meas);

        *it = L.meas_idx;

        // Sanity check on chi2
        assert(L.chi2 < std::numeric_limits<traccc::scalar>::max());
        assert(L.chi2 >= 0.f);

        ndf_sum += static_cast<scalar>(measurements.at(*it).meas_dim);
        chi2_sum += L.chi2;

        // Break the loop if the iterator is at the first candidate and fill the
        // seed and track quality
        if (it == track.measurement_indices().rend() - 1) {
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
    for (unsigned int i : track.measurement_indices()) {
        for (unsigned int j : track.measurement_indices()) {
            if (i != j) {
                // TODO: Re-enable me!
                // assert(measurements.at(i).measurement_id !=
                //       measurement.at(j).measurement_id);
            }
        }
    }
#endif
}

}  // namespace traccc::device
