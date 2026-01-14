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
    const finding_config& config, vecmem::memory_resource& mr,
    const Logger& /*log*/) {

    assert(config.min_track_candidates_per_track >= 1);

    using algebra_t = typename detector_t::algebra_type;
    using scalar_t = detray::dscalar<algebra_t>;

    // Detray propagation types
    using navigator_t = detray::caching_navigator<std::add_const_t<detector_t>>;
    using stepper_t = detray::rk_stepper<bfield_t, algebra_t>;

    using actor_chain_t =
        detray::actor_chain<detray::pathlimit_aborter<scalar_t>,
                            detray::parameter_transporter<algebra_t>,
                            detray::pointwise_material_interactor<algebra_t>,
                            traccc::measurement_updater<algebra_t>,
                            detray::parameter_resetter<algebra_t>,
                            detray::momentum_aborter<scalar_t>>;

    using propagator_t =
        detray::propagator<stepper_t, navigator_t, actor_chain_t>;

    // Create the measurement container.
    typename edm::measurement_collection<algebra_t>::const_device measurements{
        measurements_view};

    // Check contiguity of the measurements
    assert(is_contiguous_on([](const auto& value) { return value; },
                            measurements.surface_link()));

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
    const std::size_t n_seeds{seeds.size()};

    // Number of found tracks = number of seeds
    typename edm::track_container<algebra_t>::host track_container{
        mr, measurements_view};
    // Total number of tracks in one event
    track_container.tracks.reserve(n_seeds);
    // Total number of track states for all tracks in one event
    track_container.states.reserve(n_seeds *
                                   config.max_track_candidates_per_track);

    // Give the device container to the to the measurement updater
    typename edm::track_state_collection<algebra_t>::device track_states(
        vecmem::get_data(track_container.states));

    // Create detray propagator
    auto prop_cfg{config.propagation};
    propagator_t propagator(prop_cfg);

    // Configuration for measurement calibration
    measurement_selector::config calib_cfg{};

    for (unsigned int seed_idx = 0u; seed_idx < seeds.size(); ++seed_idx) {
        const auto& seed = seeds[seed_idx];

        // Construct propagation state
        typename propagator_t::state propagation(seed, field, det);
        propagation.set_particle(
            detail::correct_particle_hypothesis(config.ptc_hypothesis, seed));

        // Pathlimit aborter
        typename detray::pathlimit_aborter<scalar_t>::state aborter_state;
        // Track parameter transporter
        typename detray::parameter_transporter<algebra_t>::state
            transporter_state;
        // Material interactor
        typename detray::pointwise_material_interactor<algebra_t>::state
            interactor_state;
        // Do the measurement selection
        typename traccc::measurement_updater<algebra_t>::state
            meas_updater_state{measurements, vecmem::get_data(meas_ranges),
                               track_states};
        // Set bound track parameters after Kalman and material updates
        typename detray::parameter_resetter<algebra_t>::state resetter_state{
            prop_cfg};
        // Momentum aborter
        typename detray::momentum_aborter<scalar_t>::state
            momentum_aborter_state{};

        // Update the actor config
        momentum_aborter_state.min_pT(static_cast<scalar_t>(config.min_pT));
        momentum_aborter_state.min_p(static_cast<scalar_t>(config.min_p));

        meas_updater_state.max_chi2 = config.chi2_max;
        meas_updater_state.max_n_holes =
            static_cast<unsigned short>(config.max_num_skipping_per_cand);
        meas_updater_state.max_n_consecutive_holes =
            static_cast<unsigned short>(config.max_num_consecutive_skipped);
        meas_updater_state.m_calib_cfg = calib_cfg;

        // Current number of track states before new track is added
        const unsigned int track_states_offset{track_states.size()};

        auto actor_states = detray::tie(aborter_state, transporter_state,
                                        interactor_state, meas_updater_state,
                                        resetter_state, momentum_aborter_state);

        // Propagate the entire track
        propagator.propagate(propagation, actor_states);

        // Observe minimum track length
        const unsigned int n_new_track_states{track_states.size() -
                                              track_states_offset};
        if (n_new_track_states < config.min_track_candidates_per_track) {
            continue;
        }

        // Link the new states to the final track
        vecmem::vector<edm::track_constituent_link> state_links;
        state_links.reserve(n_new_track_states);
        for (unsigned int state_idx = track_states_offset;
             state_idx < track_states.size(); state_idx++) {
            state_links.emplace_back(edm::track_constituent_link::track_state,
                                     state_idx);
        }

        // Check track stats and build the new track object
        const track_stats<scalar_t>& trk_stats = meas_updater_state.m_stats;
        const scalar_t ndf_sum{trk_stats.ndf_sum - 5.f};

        if (ndf_sum < 0.f) {
            TRACCC_ERROR_HOST("Negative chi2 sum for track");
            continue;
        }

        // Initial track state
        track_container.tracks.push_back({});
        auto track =
            track_container.tracks.at(track_container.tracks.size() - 1u);
        track.fit_outcome() = track_fit_outcome::UNKNOWN;
        track.params() = seed;
        track.ndf() = ndf_sum;
        track.chi2() = trk_stats.chi2_sum;
        track.pval() = prob(trk_stats.chi2_sum, ndf_sum);
        track.nholes() = static_cast<unsigned int>(trk_stats.n_holes);
        track.constituent_links() = std::move(state_links);
    }

    return track_container;
}

}  // namespace traccc::host::details
