/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/track_constituent_link.hpp"
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
    const bound_track_parameters<algebra_t>& seed = seeds.at(globalIndex);

    // Access to measurements and index ranges per surface
    typename edm::measurement_collection::const_device measurements(
        payload.measurements_view);

    // Collect the track statistics
    vecmem::device_vector<track_stats<scalar_t>> track_stats(
        payload.track_stats_view);

    // Set the data pointer to the beginning of the range of the track
    const auto cand_offset{
        static_cast<int>(globalIndex * cfg.max_track_candidates_per_track)};

    track_state_candidate_data<algebra_t> candidate_data(
        cfg.run_smoother, cand_offset, payload.track_cand_view,
        payload.filtered_track_cand_view, payload.full_track_cand_view);

    // Output tracks and track state collection
    typename edm::track_collection<algebra_t>::device track_candidates(
        payload.tracks_view.tracks);
    typename edm::track_state_collection<algebra_t>::device track_states(
        payload.tracks_view.states);

    const auto ptc_hypothesis{
        detail::correct_particle_hypothesis(cfg.ptc_hypothesis, seed)};

    // Check if the seed should be forwarded to the KF
    const scalar_t q{ptc_hypothesis.charge()};
    if (seed.pT(q) <= static_cast<scalar_t>(cfg.min_pT)) {
        TRACCC_WARNING_DEVICE(
            "Seed below min. transverse momentum: |pT| = %f MeV", seed.pT(q));
        return;
    }
    if (seed.p(q) <= static_cast<scalar_t>(cfg.min_p)) {
        TRACCC_WARNING_DEVICE("Seed below min. momentum: |p| = %f MeV",
                              seed.p(q));
        return;
    }

    // Configuration for measurement calibration
    measurement_selector::config calib_cfg{};

    // Create propagator
    auto prop_cfg{cfg.propagation};
    prop_cfg.navigation.estimate_scattering_noise = false;
    propagator_t propagator(prop_cfg);

    // Create propagator state
    typename propagator_t::state propagation(seed, payload.field_data, det);
    propagation.set_particle(ptc_hypothesis);

    // Pathlimit aborter
    typename detray::actor::pathlimit_aborter<scalar_t>::state aborter_state;
    // Track parameter transporter
    typename detray::actor::parameter_updater_state<algebra_t> updater_state{
        prop_cfg, seed};
    // Material interactor
    typename detray::actor::pointwise_material_interactor<algebra_t>::state
        interactor_state;
    // Do the measurement selection
    typename traccc::measurement_updater<algebra_t>::state meas_updater_state{
        measurements, payload.measurement_ranges_view, candidate_data.ptr(),
        cfg.run_smoother};

    meas_updater_state.max_chi2 = cfg.chi2_max;
    meas_updater_state.max_n_holes =
        static_cast<unsigned short>(cfg.max_num_skipping_per_cand);
    meas_updater_state.max_n_consecutive_holes =
        static_cast<unsigned short>(cfg.max_num_consecutive_skipped);
    meas_updater_state.n_track_states_until_pause =
        static_cast<unsigned short>(cfg.duplicate_removal_minimum_length);
    meas_updater_state.m_calib_cfg = calib_cfg;

    auto actor_states = detray::tie(aborter_state, updater_state,
                                    interactor_state, meas_updater_state);

    for (unsigned int step = 0u;
         step < seeds.size() / cfg.duplicate_removal_minimum_length; ++step) {

        if (propagator.is_paused(propagation)) {
            propagator.resume(propagation);
        }

        assert(meas_updater_state.m_stats.n_holes <
               cfg.max_num_skipping_per_cand);
        assert(meas_updater_state.m_stats.n_consecutive_holes <
               cfg.max_num_consecutive_skipped);

        updater_state.notify_on_initial(step == 0u);
        propagator.propagate(propagation, actor_states);

        // Stop propagation
        if (propagator.finished(propagation) ||
            !propagator.is_paused(propagation)) {
            break;
        }

        // Check if the track should be continued
        const auto& free_param = propagation.stepping()();
        const scalar_t q_new{
            propagation.stepping().particle_hypothesis().charge()};
        if (free_param.pT(q_new) <= static_cast<scalar_t>(cfg.min_pT)) {
            TRACCC_WARNING_HOST_DEVICE(
                "Track below min. transverse momentum: |pT| = %f MeV",
                free_param.pT(q_new));
            break;
        }
        if (free_param.p(q_new) <= static_cast<scalar_t>(cfg.min_p)) {
            TRACCC_WARNING_HOST_DEVICE(
                "Track below min. momentum: |p| = %f MeV", free_param.p(q_new));
            break;
        }

        // Run track deduplication
    }

    // TODO: Removing tracks with propagation failure can get the fake rate down
    // (e.g. if it triggered an aborter)
    bool is_alive{propagator.finished(propagation)};
    if (!is_alive) {
        TRACCC_ERROR_DEVICE("Propagation failure! Track: %d", globalIndex);
    }

    // Observe minimum track length
    const traccc::track_stats<scalar_t>& trk_stats = meas_updater_state.m_stats;
    const unsigned int n_track_states{trk_stats.n_track_states};

    assert(n_track_states <= cfg.max_track_candidates_per_track);
    if (n_track_states < cfg.min_track_candidates_per_track) {
        TRACCC_WARNING_DEVICE("Short track (%d track states): discarding",
                              n_track_states);
        is_alive = false;
    }

    // Check track stats and build the new track object
    const int ndf_sum{static_cast<int>(trk_stats.ndf_sum) - 5};

    if (ndf_sum < 0) {
        TRACCC_ERROR_DEVICE("Negative NDF sum for track");
        is_alive = false;
    }

    // Link the new states to the final track
    assert(globalIndex < track_candidates.size());

    if (is_alive && trk_stats.n_holes > cfg.max_num_skipping_per_cand + 1) {
        printf("Incorrect hole count!\n");
        is_alive = false;
    }
    if (is_alive &&
        trk_stats.n_consecutive_holes > cfg.max_num_consecutive_skipped + 1) {
        printf("Incorrect consecutive hole count!\n");
        is_alive = false;
    }

    edm::track track = track_candidates.at(globalIndex);
    // Initial track state
    track.fit_outcome() =
        (is_alive && cfg.run_smoother != smoother_type::e_none)
            ? track_fit_outcome::SUCCESS
            : track_fit_outcome::UNKNOWN;
    track.params() = seed;
    track.ndf() = static_cast<scalar_t>(ndf_sum);
    track.chi2() = trk_stats.chi2_sum;
    track.pval() = prob(trk_stats.chi2_sum, static_cast<scalar_t>(ndf_sum));
    track.nholes() = static_cast<unsigned int>(trk_stats.n_holes);
    track.constituent_links().resize(is_alive ? n_track_states : 0u);

    assert(!is_alive || (n_track_states <= track.constituent_links().size()));

    if (is_alive) {
        const auto track_state_offset{globalIndex *
                                      cfg.max_track_candidates_per_track};

        for (unsigned int state_idx = track_state_offset;
             state_idx < track_state_offset + n_track_states; state_idx++) {

            const unsigned int link_idx{state_idx - track_state_offset};

            TRACCC_DEBUG_DEVICE(
                "Adding track state (local idx %d, global idx %d)", link_idx,
                state_idx);

            // Intermediate type required to build a view
            traccc::track_state_from_candidate<algebra_t>(
                candidate_data.ptr(), cfg.run_smoother, link_idx, measurements,
                track, payload.tracks_view);
        }
    }

    TRACCC_DEBUG_DEVICE("Added track %d to track container", globalIndex);
}

}  // namespace traccc::device
