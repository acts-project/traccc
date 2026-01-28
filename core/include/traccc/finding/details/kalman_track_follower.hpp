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
#include "traccc/finding/details/kalman_track_follower_types.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/finding/measurement_selector.hpp"
#include "traccc/finding/track_state_candidate.hpp"
#include "traccc/sanity/contiguous_on.hpp"
#include "traccc/utils/logging.hpp"
#include "traccc/utils/particle.hpp"
#include "traccc/utils/prob.hpp"
#include "traccc/utils/propagation.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>

namespace traccc::host::details {

/// Templated implementation of a Kalman Filter based track following algorithm.
///
/// Concrete track finding algorithms can use this function with the appropriate
/// specializations, to find tracks on top of a specific detector type, magnetic
/// field type, and track finding configuration.
///
/// @tparam detector_t The (host) detector type to use
/// @tparam bfield_t   The magnetic field type to use
///
/// @param det               The detector object
/// @param field             The magnetic field object
/// @param measurements_view All measurements in an event
/// @param seeds_view        All seeds in an event to start the track finding
///                          with
/// @param config            The track finding configuration
/// @param mr                The memory resource to use
/// @param log               The logger object to use
///
/// @return A container of the found tracks
///
template <typename detector_t, typename bfield_t>
edm::track_container<typename detector_t::algebra_type>::host
kalman_track_follower(
    const detector_t& det, const bfield_t& field,
    const typename edm::measurement_collection<
        typename detector_t::algebra_type>::const_view& measurements_view,
    const bound_track_parameters_collection_types::const_view& seeds_view,
    const finding_config& cfg, vecmem::memory_resource& mr,
    const Logger& /*log*/) {

    assert(cfg.min_track_candidates_per_track >= 1);

    using algebra_t = typename detector_t::algebra_type;
    using scalar_t = detray::dscalar<algebra_t>;

    // Detray propagation types
    using propagator_t = traccc::details::kf_propagator_t<detector_t, bfield_t>;

    // Create the measurement container.
    typename edm::measurement_collection<algebra_t>::const_device measurements{
        measurements_view};

    // Check contiguity of the measurements
    assert(is_contiguous_on([](const auto& value) { return value; },
                            measurements.surface_link()));

    TRACCC_INFO_HOST("Run Track Finding: Kalman follower");

    // Get index ranges in the measurement container per detector surface
    std::vector<unsigned int> meas_ranges;
    meas_ranges.reserve(det.surfaces().size());

    for (const auto& sf_desc : det.surfaces()) {
        // Measurements can only be found on sensitive surfaces
        if (!sf_desc.is_sensitive()) {
            // Lower range index is the upper index of the previous range
            // This is guaranteed by the measurement sorting step
            const auto sf_idx{sf_desc.index()};
            const unsigned int lo{sf_idx == 0u ? 0u : meas_ranges[sf_idx - 1u]};

            // Hand the upper index of the previous range through to assign
            // the lower index of the next valid range correctly
            meas_ranges.push_back(lo);
            continue;
        }

        auto up = std::upper_bound(measurements.surface_link().begin(),
                                   measurements.surface_link().end(),
                                   sf_desc.barcode());
        meas_ranges.push_back(static_cast<unsigned int>(
            std::distance(measurements.surface_link().begin(), up)));
    }

    // Create the input seeds container.
    bound_track_parameters_collection_types::const_device seeds{seeds_view};
    const unsigned int n_seeds{seeds.size()};

    // Number of found tracks = number of seeds
    const unsigned int max_cands{n_seeds * cfg.max_track_candidates_per_track};

    typename edm::track_container<algebra_t>::host track_container{
        mr, measurements_view};
    // Total number of tracks in one event
    track_container.tracks.reserve(n_seeds);
    // Total number of track states for all tracks in one event
    track_container.states.reserve(max_cands);

    // Track data collected by the measurement updater during pattern recog.
    vecmem::vector<track_state_candidate> track_cands{};
    vecmem::vector<filtered_track_state_candidate<algebra_t>>
        filtered_track_cands{};
    vecmem::vector<full_track_state_candidate<algebra_t>> full_track_cands{};

    switch (cfg.run_smoother) {
        case smoother_type::e_none: {
            track_cands.resize(max_cands);
            break;
        }
        case smoother_type::e_kalman: {
            filtered_track_cands.resize(max_cands);
            break;
        }
        case smoother_type::e_mbf: {
            full_track_cands.resize(max_cands);
            break;
        }
        default: {
            TRACCC_FATAL_HOST("Unknown smoother option");
            return track_container;
        }
    }

    // Create detray propagator
    auto prop_cfg{cfg.propagation};
    prop_cfg.navigation.estimate_scattering_noise = false;
    propagator_t propagator(prop_cfg);

    // Configuration for measurement calibration
    measurement_selector::config calib_cfg{};

    for (unsigned int seed_idx = 0u; seed_idx < seeds.size(); ++seed_idx) {
        const auto& seed = seeds[seed_idx];

        TRACCC_VERBOSE_HOST("Track: " << seed_idx);

        const auto ptc_hypothesis{
            detail::correct_particle_hypothesis(cfg.ptc_hypothesis, seed)};

        // Check if the seed should be forwarded to the KF
        const scalar_t q{ptc_hypothesis.charge()};
        if (seed.pT(q) <= static_cast<scalar_t>(cfg.min_pT) / 10.f) {
            TRACCC_WARNING_HOST_DEVICE(
                "Seed below min. transverse momentum: |pT| = %f MeV",
                seed.pT(q));
            continue;
        }
        if (seed.p(q) <= static_cast<scalar_t>(cfg.min_p) / 10.f) {
            TRACCC_WARNING_HOST_DEVICE("Seed below min. momentum: |p| = %f MeV",
                                       seed.p(q));
            continue;
        }

        // Set the data pointer to the beginning of the range of the track
        const auto cand_offset{static_cast<unsigned int>(
            seed_idx * cfg.max_track_candidates_per_track)};

        track_state_candidate_data<algebra_t> candidate_data(
            cfg.run_smoother, cand_offset, vecmem::get_data(track_cands),
            vecmem::get_data(filtered_track_cands),
            vecmem::get_data(full_track_cands));

        // Construct propagation state
        typename propagator_t::state propagation(seed, field, det);
        propagation.set_particle(
            detail::correct_particle_hypothesis(cfg.ptc_hypothesis, seed));

        // Pathlimit aborter
        typename detray::actor::pathlimit_aborter<scalar_t>::state
            aborter_state;
        // Track parameter transporter
        typename detray::actor::parameter_updater_state<algebra_t>
            updater_state{prop_cfg, seed};
        // Material interactor
        typename detray::actor::pointwise_material_interactor<algebra_t>::state
            interactor_state;
        // Do the measurement selection
        typename traccc::measurement_updater<algebra_t>::state
            meas_updater_state{measurements, vecmem::get_data(meas_ranges),
                               candidate_data.ptr(), cfg.run_smoother};

        meas_updater_state.max_chi2 = cfg.chi2_max;
        meas_updater_state.max_n_track_states =
            cfg.max_track_candidates_per_track;
        meas_updater_state.max_n_holes =
            static_cast<unsigned short>(cfg.max_num_skipping_per_cand);
        meas_updater_state.max_n_consecutive_holes =
            static_cast<unsigned short>(cfg.max_num_consecutive_skipped);
        meas_updater_state.n_track_states_until_pause =
            static_cast<unsigned short>(cfg.duplicate_removal_minimum_length);
        meas_updater_state.m_calib_cfg = calib_cfg;
        meas_updater_state.m_stats.seed_idx = seed_idx;

        auto actor_states = detray::tie(aborter_state, updater_state,
                                        interactor_state, meas_updater_state);

        for (unsigned int step = 0u;
             step < seeds.size() / cfg.duplicate_removal_minimum_length;
             ++step) {

            if (propagator.is_paused(propagation)) {
                propagator.resume(propagation);
            }

            assert(meas_updater_state.m_stats.n_holes <
                   cfg.max_num_skipping_per_cand);
            assert(meas_updater_state.m_stats.n_consecutive_holes <
                   cfg.max_num_consecutive_skipped);

            // Do not rerun measurement selection when resuming
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
                    "Track below min. momentum: |p| = %f MeV",
                    free_param.p(q_new));
                break;
            }

            // Run track deduplication
        }

        if (!propagator.finished(propagation)) {
            TRACCC_ERROR_HOST("Propagation failure! Track: " << seed_idx);
            continue;
        }

        // Observe minimum track length
        const track_stats<scalar_t>& trk_stats = meas_updater_state.m_stats;
        const unsigned int n_track_states{trk_stats.n_track_states};
        if (n_track_states < cfg.min_track_candidates_per_track) {
            TRACCC_WARNING_HOST("Short track ("
                                << n_track_states
                                << " track states): discarding");
            continue;
        }

        // Check track stats
        const int ndf_sum{static_cast<int>(trk_stats.ndf_sum) - 5};

        if (ndf_sum < 0) {
            TRACCC_ERROR_HOST("Negative NDF sum for track");
            continue;
        }

        TRACCC_VERBOSE_HOST("Found track for seed "
                            << seed_idx << ": (" << n_track_states
                            << " track states, " << trk_stats.n_holes
                            << " holes)");
        TRACCC_DEBUG_HOST("Seed params:\n" << seed);

        // Link the new states to the final track
        vecmem::vector<edm::track_constituent_link> state_links;
        state_links.reserve(n_track_states);

        track_container.tracks.push_back({});
        edm::track track =
            track_container.tracks.at(track_container.tracks.size() - 1u);

        if (trk_stats.n_holes > cfg.max_num_skipping_per_cand + 1) {
            TRACCC_ERROR_HOST("Hole count incorrect: max = "
                              << cfg.max_num_skipping_per_cand
                              << ", found = " << trk_stats.n_holes);
        }
        if (trk_stats.n_consecutive_holes >
            cfg.max_num_consecutive_skipped + 1) {
            TRACCC_ERROR_HOST("Consecutive hole count incorrect: max = "
                              << cfg.max_num_consecutive_skipped
                              << ", found = " << trk_stats.n_consecutive_holes);
        }

        track.fit_outcome() = cfg.run_smoother == smoother_type::e_none
                                  ? track_fit_outcome::UNKNOWN
                                  : track_fit_outcome::SUCCESS;
        track.params() = seed;
        track.ndf() = static_cast<scalar_t>(ndf_sum);
        track.chi2() = trk_stats.chi2_sum;
        track.pval() = prob(trk_stats.chi2_sum, static_cast<scalar_t>(ndf_sum));
        track.nholes() = static_cast<unsigned int>(trk_stats.n_holes);
        track.constituent_links().resize(n_track_states);

        const auto track_state_offset{
            static_cast<unsigned int>(track_container.states.size())};

        for (unsigned int state_idx = track_state_offset;
             state_idx < track_state_offset + n_track_states; state_idx++) {

            const unsigned int link_idx{state_idx - track_state_offset};

            TRACCC_DEBUG_HOST_DEVICE(
                "Adding track state (local idx %d, global idx %d)", link_idx,
                state_idx);

            // Intermediate type required to build a view
            typename edm::track_state_collection<algebra_t>::data state_data{
                vecmem::get_data(track_container.states)};
            traccc::track_state_from_candidate<algebra_t>(
                candidate_data.ptr(), cfg.run_smoother, link_idx, measurements,
                track, {state_data});
        }

        TRACCC_DEBUG_HOST("Added track " << track_container.tracks.size() - 1
                                         << " to track container");
    }

    return track_container;
}

}  // namespace traccc::host::details
