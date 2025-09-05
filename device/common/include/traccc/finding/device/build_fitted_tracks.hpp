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
#include "traccc/edm/track_fit_container.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/edm/track_state_helpers.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/utils/prob.hpp"

// VecMem include(s).
#include <limits>
#include <vecmem/containers/data/jagged_vector_view.hpp>
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Event Data) Payload for the @c traccc::device::build_tracks function
struct build_fitted_tracks_payload {
    /**
     * @brief View object to the vector of measurements
     */
    bound_track_parameters_collection_types::const_view seeds_view;

    /**
     * @brief View object to the vector of candidate links
     */
    vecmem::data::vector_view<const candidate_link> links_view;

    /**
     * @brief View object to the track parameters
     */
    bound_track_parameters_collection_types::const_view track_param_view;

    /**
     * @brief View object to the vector of tips
     */
    vecmem::data::vector_view<const unsigned int> tips_view;

    vecmem::data::vector_view<detray::geometry::barcode> barcode_sequence_view;
    vecmem::data::vector_view<unsigned int> barcode_sequence_length_view;

    /**
     * @brief View object to the vector of track candidates
     */
    edm::track_fit_container<default_algebra>::view track_fit_view;
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
TRACCC_HOST_DEVICE inline void build_fitted_tracks(
    global_index_t globalIndex, const build_fitted_tracks_payload& payload) {

    const measurement_collection_types::const_device measurements(
        payload.track_fit_view.measurements);

    const bound_track_parameters_collection_types::const_device seeds(
        payload.seeds_view);

    const vecmem::device_vector<const candidate_link> links(payload.links_view);

    const bound_track_parameters_collection_types::const_device track_params(
        payload.track_param_view);

    const vecmem::device_vector<const unsigned int> tips(payload.tips_view);

    const vecmem::device_vector<const detray::geometry::barcode>
        barcode_sequences(payload.barcode_sequence_view);
    const vecmem::device_vector<const unsigned int> barcode_sequence_lengths(
        payload.barcode_sequence_length_view);

    edm::track_fit_collection<default_algebra>::device track_candidates(
        payload.track_fit_view.tracks);

    edm::track_state_collection<default_algebra>::device track_states(
        payload.track_fit_view.states);

    if (globalIndex >= tips.size()) {
        return;
    }

    const auto tip = tips.at(globalIndex);
    edm::track_fit_collection<default_algebra>::device::proxy_type track =
        track_candidates.at(globalIndex);

    // Get the link corresponding to tip
    unsigned int L_idx = tip;
    auto L = links.at(L_idx);

    unsigned int total_states = 0;

    while (true) {
        if (L.step == 0) {
            assert(barcode_sequence_lengths.at(L_idx) == 0);
            total_states += 1;
            break;
        } else {
            total_states +=
                std::max(1u, barcode_sequence_lengths.at(L_idx) - 1u);
            L_idx = L.previous_candidate_idx;
            L = links.at(L_idx);
        }
    }

    L_idx = tip;
    L = links.at(L_idx);

    const unsigned int n_meas = measurements.size();

    // Track summary variables
    scalar ndf_sum = 0.f;
    scalar chi2_sum = 0.f;

    track.state_indices().resize(total_states);
    track.barcodes().resize(total_states);

    unsigned int state_idx = total_states - 1;

    // Reversely iterate to fill the track candidates
    while (true) {

        if (L.meas_idx < measurements.size()) {
            const unsigned int track_state_index =
                track_states.push_back(edm::make_track_state<default_algebra>(
                    measurements, L.meas_idx));

            track.barcodes().at(state_idx) =
                track_params.at(L_idx).surface_link();
            track.state_indices().at(state_idx--) = track_state_index;

            ndf_sum +=
                static_cast<scalar>(measurements.at(L.meas_idx).meas_dim);
            chi2_sum += L.chi2;

            track_states.filtered_chi2().at(track_state_index) = L.chi2;
            track_states.filtered_params().at(track_state_index) =
                track_params.at(L_idx);
        } else {
            track.barcodes().at(state_idx) = barcode_sequences.at(
                10 * L_idx + barcode_sequence_lengths.at(L_idx) - 1);
            track.state_indices().at(state_idx--) =
                std::numeric_limits<unsigned int>::max();
        }

        for (unsigned int i = 1; i + 1 < barcode_sequence_lengths.at(L_idx);
             ++i) {
            track.barcodes().at(state_idx) = barcode_sequences.at(
                10 * L_idx + (barcode_sequence_lengths.at(L_idx) - (i + 1)));
            track.state_indices().at(state_idx--) =
                std::numeric_limits<unsigned int>::max();
        }

        // Break the loop if the iterator is at the first candidate and fill the
        // seed and track quality
        if (L.step == 0) {
            assert(state_idx >= total_states);
            track.fit_outcome() = track_fit_outcome::SUCCESS;
            track.params() = seeds.at(L.seed_idx);
            track.ndf() = ndf_sum - 5.f;
            track.chi2() = chi2_sum;
            track.pval() = prob(track.chi2(), track.ndf());
            track.nholes() = L.n_skipped;
            break;
        } else {
            L_idx = L.previous_candidate_idx;
            L = links.at(L_idx);
        }
    }

#ifndef NDEBUG
    // Assert that we did not make any duplicate track states.
    for (unsigned int i : track.state_indices()) {
        for (unsigned int j : track.state_indices()) {
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
