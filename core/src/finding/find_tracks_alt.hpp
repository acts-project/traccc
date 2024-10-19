/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/fitting/kalman_filter/gain_matrix_updater.hpp"
#include "traccc/sanity/contiguous_on.hpp"
#include "traccc/utils/particle.hpp"
#include "traccc/utils/projections.hpp"

// Detray include(s).
// #include <detray/definitions/detail/algorithms.hpp> // < detray::upper_bound
#include <detray/propagator/actor_chain.hpp>
#include <detray/propagator/actors/aborters.hpp>
#include <detray/propagator/actors/parameter_resetter.hpp>
#include <detray/propagator/actors/parameter_transporter.hpp>
#include <detray/propagator/actors/pointwise_material_interactor.hpp>
#include <detray/propagator/propagator.hpp>

// System include
#include <algorithm>

namespace traccc::details {

template <typename stepper_t, typename navigator_t>
track_candidate_container_types::host find_tracks(
    const typename navigator_t::detector_type& det,
    const typename stepper_t::magnetic_field_type& field,
    const measurement_collection_types::const_view& measurements_view,
    const bound_track_parameters_collection_types::const_view& seeds_view,
    const finding_config& config) {

    /*****************************************************************
     * Types used by the track finding
     *****************************************************************/

    /// Algebra types
    using algebra_t = typename navigator_t::detector_type::algebra_type;

    /// Actor types
    using aborter = detray::pathlimit_aborter;
    using transporter = detray::parameter_transporter<algebra_t>;
    using interactor = detray::pointwise_material_interactor<algebra_t>;
    using resetter = detray::parameter_resetter<algebra_t>;

    using actor_type = detray::actor_chain<detray::tuple, aborter, transporter,
                                           interactor, resetter>;

    using propagator_type =
        detray::propagator<stepper_t, navigator_t, actor_type>;

    /// Helper structs
    /// @{
    /// Navigation state and current bound track params for a track candidate
    struct navigation_stream {
        typename navigator_t::state state;
        bound_track_parameters track_params;
    };

    /// Trace the measurements and number of holes per track
    struct trace_state {
        // Track index that this candidate was branched off from
        unsigned int parent_idx{std::numeric_limits<unsigned int>::max()};
        // Index of measurement in original track where this cand. branched off
        unsigned int branch_meas_idx{0u};
        // Number of surfaces reached without compatible measurement
        unsigned int n_skipped{0u};
        // Number of total measurements assigned to a track candidate
        unsigned int n_meas{0u};
        // Indices of measurements after initial branching
        vecmem::vector<unsigned int> meas_indices{};
    };

    /// Package measurements with the corresponding filtered track parameters
    struct candidate {
        bound_track_parameters filtered_params;
        unsigned int meas_idx;
        float chi2;

        /// @param rhs is the right hand side candidate for comparison
        constexpr bool operator<(const candidate& rhs) const {
            return (chi2 < rhs.chi2);
        }

        /// @param rhs is the left hand side candidate for comparison
        constexpr bool operator>(const candidate& rhs) const {
            return (chi2 > rhs.chi2);
        }
    };
    /// @}

    /*****************************************************************
     * Measurement Operations
     *****************************************************************/

    // Create the measurement container.
    measurement_collection_types::const_device measurements{measurements_view};

    // Check contiguity of the measurements
    assert(
        host::is_contiguous_on(measurement_module_projection(), measurements));

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

        auto up = std::upper_bound(measurements.begin(), measurements.end(),
                                   sf_desc.barcode(), measurement_bcd_comp());
        meas_ranges.push_back(
            static_cast<unsigned int>(std::distance(measurements.begin(), up)));
    }

    /*****************************************************************
     * CKF Preparations
     *****************************************************************/

    // Create the input seeds container.
    bound_track_parameters_collection_types::const_device seeds{seeds_view};

    const std::size_t n_seeds{seeds.size()};
    const unsigned int n_max_nav_streams{
        math::min(config.max_num_branches_per_seed, 20000u)};
    const unsigned int n_max_branches_per_surface{
        math::min(config.max_num_branches_per_surface, 10u)};

    // Measurement trace per track
    std::vector<trace_state> traces;
    traces.resize(n_seeds * n_max_nav_streams);

    // Compatible measurements and filtered track params on a given surface
    std::vector<candidate> candidates;
    candidates.reserve(n_max_branches_per_surface);

    // Create detray propagator
    propagator_type propagator(config.propagation);

    // Navigation streams: One inner vector per seed, which contains
    // the branched navigation streams for the respective seed
    std::vector<std::vector<navigation_stream>> nav_streams_per_seed;
    nav_streams_per_seed.reserve(n_seeds);

    // Create initial navigation stream for each seed
    for (const auto& seed : seeds) {
        auto& nav_streams = nav_streams_per_seed.emplace_back();
        nav_streams.reserve(n_max_nav_streams);

        auto& nav_stream =
            nav_streams.emplace_back(typename navigator_t::state(det), seed);
        nav_stream.state.set_volume(seed.surface_link().volume());

        // Construct propagation state around the navigation stream
        typename propagator_type::non_owning_state propagation(
            nav_stream.track_params, field, nav_stream.state);
        propagation.set_particle(detail::correct_particle_hypothesis(
            config.ptc_hypothesis, nav_stream.track_params));

        // Create actor states
        typename aborter::state s0{config.propagation.stepping.path_limit};
        typename transporter::state s1;
        typename interactor::state s2;
        typename resetter::state s3;

        auto actor_states = detray::tie(s0, s1, s2, s3);

        propagator.propagate_init(propagation, actor_states);

        // Make sure the CKF can start on a sensitive surface
        if (propagation._heartbeat && !nav_stream.state.is_on_sensitive()) {
            propagator.propagate_to_next(propagation, actor_states);
        }
        // Either exited detector by portal right away or are on first
        // sensitive surface
        assert(nav_stream.state.is_complete() ||
               nav_stream.state.is_on_sensitive());
    }

    // Step through the sensitive surfaces along the tracks
    const auto n_steps{static_cast<int>(config.max_track_candidates_per_track)};
    for (int step = 0; step < n_steps; step++) {
        // Step through all track candidates (branches) for a given seed
        for (unsigned int seed_idx = 0u; seed_idx < n_seeds; seed_idx++) {

            auto& nav_streams = nav_streams_per_seed[seed_idx];
            const auto n_traces{static_cast<unsigned int>(nav_streams.size())};
            assert(n_traces >= 1u);
            for (unsigned int br_idx = 0u; br_idx < n_traces; ++br_idx) {
                // The navigation stream for this trace
                auto& nav_stream = nav_streams[br_idx];
                const auto& navigation = nav_stream.state;

                // Propagation is no longer alive (trace finished or hit error)
                if (!navigation.is_alive()) {
                    continue;
                }
                assert(navigation.is_on_sensitive());

                // Get current detector surface (sensitive)
                const auto sf = navigation.get_surface();

                /***************************************************************
                 * Find compatible measurements
                 **************************************************************/

                // Iterate over the measurements for this surface
                const auto sf_idx{sf.index()};
                const unsigned int lo{sf_idx == 0u ? 0u
                                                   : meas_ranges[sf_idx - 1]};
                const unsigned int up{meas_ranges[sf_idx]};

                for (unsigned int meas_id = lo; meas_id < up; meas_id++) {

                    track_state<algebra_t> trk_state(measurements[meas_id]);

                    // Run the Kalman update on a copy of the track parameters
                    const bool res =
                        sf.template visit_mask<gain_matrix_updater<algebra_t>>(
                            trk_state, nav_stream.track_params);
                    // Found a good measurement?
                    if (const auto chi2 = trk_state.filtered_chi2();
                        res && chi2 < config.chi2_max) {
                        candidates.emplace_back(trk_state.filtered(), meas_id,
                                                chi2);
                    }
                }

                /***************************************************************
                 * Update current navigation stream and branch
                 **************************************************************/

                const unsigned int trc_idx{seed_idx * n_max_nav_streams +
                                           br_idx};
                auto& parent = traces[trc_idx];

                // Count hole in case no measurements were found
                if (candidates.empty()) {
                    parent.n_skipped++;

                    // If number of skips is larger than the maximal value,
                    // consider the track to be finished
                    if (parent.n_skipped > config.max_num_skipping_per_cand) {
                        nav_stream.state.abort();
                        assert(!navigation.is_alive());
                    }
                } else {
                    // Consider only the best candidates
                    std::sort(candidates.begin(), candidates.end());

                    // Update the track parameters in the current navigation
                    nav_stream.track_params = candidates[0u].filtered_params;
                    parent.meas_indices.push_back(candidates[0u].meas_idx);
                    parent.n_meas++;
                    // @TODO: Make truslevel setting configurable
                    nav_stream.state.set_high_trust();

                    // Number of potential new branches
                    unsigned int n_branches{math::min(
                        static_cast<unsigned int>(candidates.size()) - 1u,
                        n_max_branches_per_surface)};

                    // Number of allowed new branches for this seed
                    auto allowed_branches{static_cast<int>(n_max_nav_streams) -
                                          static_cast<int>(nav_streams.size())};
                    allowed_branches =
                        math::signbit(allowed_branches) ? 0 : allowed_branches;

                    // Create new branches
                    n_branches =
                        math::min(n_branches,
                                  static_cast<unsigned int>(allowed_branches));
                    for (unsigned int i = 0u; i < n_branches; ++i) {
                        // Clone current navigation stream for new branch with
                        // updated track parameters
                        nav_streams.emplace_back(
                            nav_stream.state,
                            candidates[i + 1u].filtered_params);

                        // Get the measurement trace for the new branch
                        const std::size_t branch_idx{nav_streams.size() - 1u};
                        auto& branch =
                            traces[seed_idx * n_max_nav_streams + branch_idx];

                        // Copy metadata from original track candidate to branch
                        branch.parent_idx = br_idx;
                        branch.branch_meas_idx = parent.n_meas - 1u;
                        branch.n_meas = parent.n_meas;
                        branch.n_skipped = parent.n_skipped;

                        // Add new measurement
                        branch.meas_indices.reserve(
                            config.max_track_candidates_per_track -
                            branch.n_meas);
                        branch.meas_indices.push_back(
                            candidates[i + 1u].meas_idx);
                    }
                }
                candidates.clear();
            }

            /*******************************************************
             * Propagate all tracks of this seed to the next surface
             *******************************************************/
            for (auto& nav_stream : nav_streams) {
                // Propagation is no longer alive (track finished or hit error)
                if (!nav_stream.state.is_alive()) {
                    continue;
                }
                assert(nav_stream.state.is_on_sensitive());

                typename propagator_type::non_owning_state propagation(
                    nav_stream.track_params, field, nav_stream.state);
                propagation.set_particle(detail::correct_particle_hypothesis(
                    config.ptc_hypothesis, nav_stream.track_params));
                // Has been initialized in the beginning
                propagation._heartbeat = nav_stream.state.is_alive();

                // Update distance to next surface for filtered track state
                navigator_t{}.update(propagation.stepping()(), nav_stream.state,
                                     config.propagation.navigation);
                propagation.stepping().set_step_size(nav_stream.state());

                assert(propagation._heartbeat);
                assert(nav_stream.state.is_alive());

                // Create actor states
                // @TODO: This does not work, the actor states need to be kept
                // alive as well
                typename aborter::state s0{
                    config.propagation.stepping.path_limit};
                typename transporter::state s1;
                typename interactor::state s2;
                typename resetter::state s3;

                // Propagate to the next surface
                propagator.propagate_to_next(propagation,
                                             detray::tie(s0, s1, s2, s3));

                assert(nav_stream.state.is_complete() ||
                       nav_stream.state.is_on_sensitive());

                // TODO: Find a way to not copy the track params all the time
                nav_stream.track_params = propagation.stepping().bound_params();
            }
        }
    }

    /**********************
     * Build tracks
     **********************/

    // @TODO: Remove once multi-trajectory container is available
    track_candidate_container_types::host output_candidates;
    output_candidates.reserve(2u * n_seeds);

    // Map the position of a branch in the output container to its index in the
    // trace vector (some branches are skipped if they are too short, in which
    // case this is not a 1:1 relation anymore)
    vecmem::vector<unsigned int> branch_to_track_map{};
    branch_to_track_map.resize(n_max_nav_streams);

    // Buffer to keep the first few measurements that were not saved in the
    // output collection in case the original branch(es) (stem) were too short
    // This can never be more than the min. number of measurements per track
    vecmem::vector<measurement> stem_buffer{};
    stem_buffer.reserve(config.min_track_candidates_per_track);

    // Step through all track candidates for a given seed
    for (unsigned int seed_idx = 0u; seed_idx < n_seeds; seed_idx++) {
        // Get the seed
        const auto seed = seeds[seed_idx];
        // Buffer the first measurements of the new seed
        stem_buffer.clear();

        const std::size_t n_branches{nav_streams_per_seed[seed_idx].size()};
        for (unsigned int br_idx = 0u; br_idx < n_branches; ++br_idx) {

            const unsigned int trc_idx{seed_idx * n_max_nav_streams + br_idx};
            const auto& trace = traces[trc_idx];
            // Reset the map for this seed as we go along (branches will
            // strictly only look for a parent in the preceeding entries)
            branch_to_track_map[br_idx] =
                std::numeric_limits<unsigned int>::max();

            // Not enough measurements found to make a track candidate, but keep
            // the measurements around to build the branches of this track
            if (trace.n_meas < config.min_track_candidates_per_track) {
                // Fill the stem buffer instead of the output collection
                for (unsigned int meas_idx : trace.meas_indices) {
                    assert(meas_idx < measurements.size());
                    stem_buffer.push_back(measurements[meas_idx]);
                    // If more measurements had accumulated, the track would be
                    // in the output collection
                    assert(stem_buffer.size() <=
                           config.min_track_candidates_per_track);
                }
                continue;
            }

            vecmem::vector<measurement> trk_measurements{};
            trk_measurements.reserve(trace.n_meas);

            // Get the relevant measurement collection from the parent
            // Later branches appear at the back of the collection, so the
            // parent is already fully assembled in the output cont./stem buffer
            if (trace.parent_idx < traces.size()) {
                std::size_t parent_trk_idx{
                    branch_to_track_map[trace.parent_idx]};

                if (parent_trk_idx < output_candidates.size()) {

                    const auto& [_, parent_trk] =
                        output_candidates.at(parent_trk_idx);

                    assert(trace.branch_meas_idx < parent_trk.size());

                    std::ranges::copy_n(std::ranges::begin(parent_trk),
                                        trace.branch_meas_idx,
                                        std::back_inserter(trk_measurements));
                } else {
                    // The branching meas. index on a short track must be small
                    assert(trace.branch_meas_idx < stem_buffer.size());

                    std::ranges::copy_n(std::ranges::begin(stem_buffer),
                                        trace.branch_meas_idx,
                                        std::back_inserter(trk_measurements));
                }
            }

            // Add the measurements collected on this branch
            for (unsigned int meas_idx : trace.meas_indices) {
                assert(meas_idx < measurements.size());
                trk_measurements.push_back(measurements[meas_idx]);
            }
            assert(trace.n_meas == trk_measurements.size());

            // All measurements have been transcribed, move to final container
            assert(br_idx < branch_to_track_map.size());
            branch_to_track_map[br_idx] = output_candidates.size();
            output_candidates.push_back(bound_track_parameters{seed},
                                        std::move(trk_measurements));
        }
    }

    return output_candidates;
}

}  // namespace traccc::details
