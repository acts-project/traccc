/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include
#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

// VecMem include(s).
#include <vecmem/containers/vector.hpp>

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/messaging.hpp"

// Greedy ambiguity resolution adapted from ACTS code

namespace traccc {

/// Evicts tracks that seem to be duplicates or fakes. This algorithm takes a
/// greedy approach in the sense that it will remove the track which looks "most
/// duplicate/fake" first and continues the same process with the rest. That
/// process continues until the final state conditions are met.
///
/// The implementation works as follows:
///  1) Calculate shared hits per track.
///  2) If the maximum shared hits criteria is met, we are done.
///     This is the configurable amount of shared hits we are ok with
///     in our experiment.
///  3) Else, remove the track with the highest relative shared hits (i.e.
///     shared hits / hits).
///  4) Back to square 1.
class greedy_ambiguity_resolution_algorithm
    : public algorithm<track_state_container_types::host(
          const typename track_state_container_types::host&)>,
      public messaging {

    public:
    struct config_t {

        config_t(){};

        /// Maximum amount of shared hits per track. One (1) means "no shared
        /// hit allowed".
        std::uint32_t maximum_shared_hits = 1;

        /// Maximum number of iterations.
        std::uint32_t maximum_iterations = 1000000;

        /// Minimum number of measurement to form a track.
        std::size_t n_measurements_min = 3;

        // True if obvious errors should be checked after the completion
        // of the algorithm.
        bool check_obvious_errs = true;

        // Displays a warning if:
        // (number of measurements of ID 0 / total number of measurements)
        // is greater than measurement_id_0_warning_threshold.
        float measurement_id_0_warning_threshold = 0.1f;

        bool verbose_error = true;
        bool verbose_warning = true;
        bool verbose_info = false;
        bool verbose_debug = false;
    };

    struct state_t {
        std::size_t number_of_tracks{};

        /// For this whole comment section, track_index refers to the index of a
        /// track in the initial input container.
        ///
        /// There is no (track_id) in this algorithm, only (track_index).

        /// Associates each track_index with the track's chi2 value
        std::vector<traccc::scalar> track_chi2;

        /// Associates each track_index to the track's (measurement_id)s list
        std::vector<std::vector<std::size_t>> measurements_per_track;

        /// Associates each measurement_id to a set of (track_index)es sharing
        /// it
        std::unordered_map<std::size_t, std::set<std::size_t>>
            tracks_per_measurement;

        /// Associates each track_index to its number of shared measurements
        /// (among other tracks)
        std::vector<std::size_t> shared_measurements_per_track;

        /// Keeps the selected tracks indexes that have not (yet) been removed
        /// by the algorithm
        std::set<std::size_t> selected_tracks;
    };

    /// Constructor for the greedy ambiguity resolution algorithm
    ///
    /// @param cfg  Configuration object
    // greedy_ambiguity_resolution_algorithm(const config_type& cfg) :
    // _config(cfg) {}
    greedy_ambiguity_resolution_algorithm(
        const config_t cfg,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone())
        : messaging(std::move(logger)), _config{cfg} {}

    /// Run the algorithm
    ///
    /// @param track_states the container of the fitted track parameters
    /// @return the container without ambiguous tracks
    track_state_container_types::host operator()(
        const typename track_state_container_types::host& track_states)
        const override;

    private:
    /// Computes the initial state for the input data. This function accumulates
    /// information that will later be used to accelerate the ambiguity
    /// resolution.
    ///
    /// @param track_states The input track container (output of the fitting
    /// algorithm).
    /// @param state An empty state object which is expected to be default
    /// constructed.
    void compute_initial_state(
        const typename track_state_container_types::host& track_states,
        state_t& state) const;

    /// Updates the state iteratively by evicting one track after the other
    /// until the final state conditions are met.
    ///
    /// @param state A state object that was previously filled by the
    /// initialization.
    void resolve(state_t& state) const;

    /// Check for obvious errors returned by the algorithm:
    /// - Returned tracks should be independent of each other: they should share
    ///   a maximum of (_config.maximum_shared_hits - 1) hits per track.
    /// - Each removed track should share at least (_config.maximum_shared_hits)
    ///   with another initial track.
    ///
    /// @param initial_track_states The input track container, as given to
    /// compute_initial_state.
    /// @param final_state The state object after the resolve method has been
    /// called.
    bool check_obvious_errors(
        const typename track_state_container_types::host& initial_track_states,
        state_t& final_state) const;

    config_t _config;
};

}  // namespace traccc
