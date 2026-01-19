/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/track_container.hpp"
#include "traccc/edm/track_state_helpers.hpp"
#include "traccc/finding/actors/measurement_kalman_updater.hpp"
#include "traccc/finding/details/combinatorial_kalman_filter_types.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/finding/measurement_selector.hpp"
#include "traccc/finding/track_state_candidate.hpp"
#include "traccc/sanity/contiguous_on.hpp"
#include "traccc/utils/logging.hpp"
#include "traccc/utils/particle.hpp"
#include "traccc/utils/prob.hpp"
#include "traccc/utils/propagation.hpp"

// Detray include(s).
#include <detray/plugins/algebra/array_definitions.hpp>
#include <detray/utils/tuple_helpers.hpp>

namespace traccc::device {

template <typename propagator_t>
TRACCC_HOST_DEVICE inline void kalman_track_follower(
    const global_index_t globalIndex, const finding_config& cfg,
    const kalman_track_follower_payload<propagator_t>& payload) {

    using detector_t = typename propagator_t::detector_type;
    using algebra_t = typename detector_t::algebra_type;
    using scalar_t = detray::dscalar<algebra_t>;

    if (globalIndex >= payload.seeds_view.size()) {
        return;
    }

    // Detector
    detector_t det(payload.det_data);

    // Access to initial bound track parameters
    bound_track_parameters_collection_types::device seeds(payload.seeds_view);

    const bound_track_parameters<algebra_t> seed = seeds.at(globalIndex);

    // Access to measurements and index ranges per surface
    typename edm::measurement_collection<algebra_t>::const_device measurements(
        payload.measurements_view);

    // Collect the track statistics
    vecmem::device_vector<track_stats<scalar_t>> track_stats(
        payload.track_stats_view);

    // Output data collections
    vecmem::device_vector<track_state_candidate> track_cands{
        payload.track_cand_view};
    vecmem::device_vector<filtered_track_state_candidate<algebra_t>>
        filtered_track_cands{payload.filtered_track_cand_view};
    vecmem::device_vector<full_track_state_candidate<algebra_t>>
        full_track_cands{payload.full_track_cand_view};

    // Number of found tracks = number of seeds
    const unsigned int max_cands{seeds.size() *
                                 cfg.max_track_candidates_per_track};
    const auto current_cand{
        static_cast<int>(globalIndex * cfg.max_track_candidates_per_track)};

    void* track_cand_ptr{nullptr};
    if (cfg.run_smoother == smoother_type::e_none) {
        assert(track_cands.size() == max_cands);
        track_cand_ptr = static_cast<void*>(
            detray::ranges::detail::next(track_cands.data(), current_cand));
    } else if (cfg.run_smoother == smoother_type::e_kalman) {
        assert(filtered_track_cands.size() == max_cands);
        track_cand_ptr = static_cast<void*>(detray::ranges::detail::next(
            filtered_track_cands.data(), current_cand));
    } else if (cfg.run_smoother == smoother_type::e_mbf) {
        assert(full_track_cands.size() == max_cands);
        track_cand_ptr = static_cast<void*>(detray::ranges::detail::next(
            full_track_cands.data(), current_cand));
    } else {
        TRACCC_ERROR_DEVICE("Unknown smoother option");
        return;
    }
    assert(track_cand_ptr);

    // Output tracks and track state collection
    /*typename edm::track_collection<algebra_t>::device track_candidates(
        payload.tracks_view.tracks);
    typename edm::track_state_collection<algebra_t>::device track_states(
        payload.tracks_view.states);*/

    // Configuration for measurement calibration
    measurement_selector::config calib_cfg{};

    // Create propagator
    auto prop_cfg{cfg.propagation};
    propagator_t propagator(prop_cfg);

    // Create propagator state
    typename propagator_t::state propagation(seed, payload.field_data, det);
    propagation.set_particle(
        detail::correct_particle_hypothesis(cfg.ptc_hypothesis, seed));

    // Pathlimit aborter
    typename detray::pathlimit_aborter<scalar_t>::state aborter_state;
    // Track parameter transporter
    typename detray::parameter_transporter<algebra_t>::state transporter_state;
    // Material interactor
    typename detray::pointwise_material_interactor<algebra_t>::state
        interactor_state;
    // Do the measurement selection
    typename traccc::measurement_updater<algebra_t>::state meas_updater_state{
        measurements, payload.measurement_ranges_view, track_cand_ptr,
        cfg.run_smoother};
    // Set bound track parameters after Kalman and material updates
    typename detray::parameter_resetter<algebra_t>::state resetter_state{
        prop_cfg};
    // Momentum aborter
    typename detray::momentum_aborter<scalar_t>::state momentum_aborter_state{};

    // Update the actor config
    momentum_aborter_state.min_pT(static_cast<scalar_t>(cfg.min_pT));
    momentum_aborter_state.min_p(static_cast<scalar_t>(cfg.min_p));

    meas_updater_state.max_chi2 = cfg.chi2_max;
    meas_updater_state.max_n_holes =
        static_cast<unsigned short>(cfg.max_num_skipping_per_cand);
    meas_updater_state.max_n_consecutive_holes =
        static_cast<unsigned short>(cfg.max_num_consecutive_skipped);
    meas_updater_state.m_calib_cfg = calib_cfg;

    auto actor_states =
        detray::tie(aborter_state, transporter_state, interactor_state,
                    meas_updater_state, resetter_state, momentum_aborter_state);

    // Propagate the entire track
    propagator.propagate(propagation, actor_states);

    // If a surface found, save the track stats to build the tracks
    if (propagator.finished(propagation) &&
        meas_updater_state.m_stats.n_track_states <
            cfg.min_track_candidates_per_track) {
        assert(propagation._navigation.is_on_sensitive());
        assert(!propagation._stepping.bound_params().is_invalid());

        track_stats.at(globalIndex) = meas_updater_state.m_stats;
    } else {
        TRACCC_ERROR_DEVICE("Propagation failure!");
    }

    /*if (is_alive) {
        // Link the new states to the final track
        auto track = track_candidates.at(globalIndex);
        for (unsigned int state_idx = track_states_offset;
             state_idx < meas_updater_state.state_idx; state_idx++) {
            track.constituent_links().push_back(
                {edm::track_constituent_link::track_state, state_idx});
        }

        // Check track stats and build the new track object
        const track_stats<scalar_t>& trk_stats = meas_updater_state.m_stats;
        const scalar_t ndf_sum{trk_stats.ndf_sum - 5.f};

        if (ndf_sum < 0.f) {
            TRACCC_ERROR_DEVICE("Negative chi2 sum for track");
            return;
        }

        // Fill the track information
        track.fit_outcome() = track_fit_outcome::UNKNOWN;
        track.params() = seed;
        track.ndf() = ndf_sum;
        track.chi2() = trk_stats.chi2_sum;
        track.pval() = prob(trk_stats.chi2_sum, ndf_sum);
        track.nholes() = static_cast<unsigned int>(trk_stats.n_holes);
    }*/
}

}  // namespace traccc::device
