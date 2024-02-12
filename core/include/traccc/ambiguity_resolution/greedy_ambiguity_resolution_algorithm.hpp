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
          const typename track_state_container_types::host&)> {

    public:
    struct config_t {
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

        bool verbose_info = true;
        bool verbose_error = true;
        bool verbose_flood = false;
    };

    struct state_t {
        std::size_t number_of_tracks{};

        /// For this whole comment section, track_index refers to the index of a
        /// track in the initial input container.
        ///
        /// There is no (track_id) in this algorithm, only (track_index).

        /// Associates each track_index with the track's chi2 value
        std::vector<float> track_chi2;

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

    void verbose_info(std::string s) const {
        if (_config.verbose_info)
            std::cout << "@greedy_ambiguity_resolution_algorithm: " << s
                      << "\n";
    }

    void verbose_error(std::string s) const {
        if (_config.verbose_error)
            std::cout << "ERROR @greedy_ambiguity_resolution_algorithm: " << s
                      << "\n";
    }

    void verbose_flood(std::string s) const {
        if (_config.verbose_flood)
            std::cout << "@greedy_ambiguity_resolution_algorithm: " << s
                      << "\n";
    }

    /// Constructor for the greedy ambiguity resolution algorithm
    ///
    /// @param cfg  Configuration object
    // greedy_ambiguity_resolution_algorithm(const config_type& cfg) :
    // _config(cfg) {}
    greedy_ambiguity_resolution_algorithm(const config_t& cfg) : _config{cfg} {}

    /// Default constructor for the greedy ambiguity resolution algorithm,
    /// with default configuration
    greedy_ambiguity_resolution_algorithm() {}

    /// Run the algorithm
    ///
    /// @param track_states the container of the fitted track parameters
    /// @return the container without ambiguous tracks
    track_state_container_types::host operator()(
        const typename track_state_container_types::host& track_states)
        const override {

        state_t state;
        compute_initial_state(track_states, state);
        resolve(state);

        if (_config.check_obvious_errs) {
            verbose_info("Checking result validity...");
            check_obvious_errors(track_states, state);
        }

        // Copy the tracks to be retained in the return value

        track_state_container_types::host res;
        res.reserve(state.selected_tracks.size());

        verbose_info("state.selected_tracks.size() = " +
                     std::to_string(state.selected_tracks.size()));

        for (std::size_t index : state.selected_tracks) {
            // track_states is a host_container<fitting_result<transform3>,
            // track_state<transform3>>
            auto const [sm_headers, sm_items] = track_states.at(index);

            // Copy header
            fitting_result<transform3> header = sm_headers;

            // Copy states
            vecmem::vector<track_state<transform3>> states;
            states.reserve(sm_items.size());
            for (auto const& item : sm_items) {
                states.push_back(item);
            }

            res.push_back(header, states);
        }
        return res;
    }

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
    /// with another initial track.
    ///
    /// @param initial_track_states The input track container, as given to
    /// compute_initial_state.
    /// @param final_state The state object after the resolve method has been
    /// called.
    bool check_obvious_errors(
        const typename track_state_container_types::host& initial_track_states,
        state_t& final_state) const;

    private:
    config_t _config;
};

// Implementation

void greedy_ambiguity_resolution_algorithm::compute_initial_state(
    const typename track_state_container_types::host& track_states,
    state_t& state) const {

    // For each track of the input container
    std::size_t n_track_states = track_states.size();
    for (std::size_t track_index = 0; track_index < n_track_states;
         ++track_index) {

        // fit_res is a fitting_result<transform3>
        // states  is a vecmem_vector<track_state<transform3>>
        auto const& [fit_res, states] = track_states.at(track_index);

        // Kick out tracks that do not fulfill our initial requirements
        if (states.size() < _config.n_measurements_min) {
            continue;
        }

        // Create the list of measurement_id of the current track
        std::vector<std::size_t> measurements;
        for (auto const& st : states) {
            measurements.push_back(st.get_measurement().measurement_id);
        }

        // Add this track chi2 value
        state.track_chi2.push_back(fit_res.chi2);
        // Add all the (measurement_id)s of this track
        state.measurements_per_track.push_back(std::move(measurements));
        // Initially, every track is in the selected_track list. They will later
        // be removed according to the algorithm.
        state.selected_tracks.insert(state.number_of_tracks);
        ++state.number_of_tracks;
    }

    // Associate each measurement to the tracks sharing it
    for (std::size_t track_index = 0; track_index < state.number_of_tracks;
         ++track_index) {
        for (auto meas_id : state.measurements_per_track[track_index]) {
            state.tracks_per_measurement[meas_id].insert(track_index);
        }
    }

    // Finally, we can accumulate the number of shared measurements per track
    state.shared_measurements_per_track =
        std::vector<std::size_t>(state.number_of_tracks, 0);

    for (std::size_t track_index = 0; track_index < state.number_of_tracks;
         ++track_index) {
        for (auto meas_index : state.measurements_per_track[track_index]) {
            if (state.tracks_per_measurement[meas_index].size() > 1) {
                ++state.shared_measurements_per_track[track_index];
            }
        }
    }
}

/// Check for obvious errors returned by the algorithm:
/// - Returned tracks should be independent of each other: they should share a
/// maximum of (_config.maximum_shared_hits - 1) hits per track.
/// - Each removed track should share at least (_config.maximum_shared_hits)
/// with another initial track.
///
/// @param initial_track_states The input track container, as given to
/// compute_initial_state.
/// @param final_state The state object after the resolve method has been
/// called.
bool greedy_ambiguity_resolution_algorithm::check_obvious_errors(
    const typename track_state_container_types::host& initial_track_states,
    state_t& final_state) const {

    // Associates every measurement_id to the number of tracks that shares it
    // (during initial state)
    std::unordered_map<std::size_t, std::size_t> initial_measurement_count;

    // Initialize initial_measurement_count
    for (std::size_t track_index = 0; track_index < initial_track_states.size();
         ++track_index) {
        // fit_res is a fitting_result<transform3>
        // states  is a vecmem_vector<track_state<transform3>>
        auto const& [fit_res, states] = initial_track_states.at(track_index);

        for (auto const& st : states) {
            std::size_t meas_id = st.get_measurement().measurement_id;

            std::unordered_map<std::size_t, std::size_t>::iterator meas_it =
                initial_measurement_count.find(meas_id);

            if (meas_it == initial_measurement_count.end()) {
                // not found: for now, this measurement only belongs to one
                // track
                initial_measurement_count[meas_id] = 1;
            } else {
                // found: this measurement is shared between at least two tracks
                ++(meas_it->second);
            }
        }
    }

    bool all_removed_tracks_alright = true;
    // ===========================================================================
    // Checks that every removed track had at least
    // (_config.maximum_shared_hits) commun measurements with other tracks
    // ===========================================================================
    for (std::size_t track_index = 0; track_index < initial_track_states.size();
         ++track_index) {
        auto const& [fit_res, states] = initial_track_states.at(track_index);

        // If the current track has been removed from selected_tracks
        if (final_state.selected_tracks.find(track_index) ==
            final_state.selected_tracks.end()) {

            std::size_t shared_hits = 0;

            for (auto const& st : states) {
                auto meas_id = st.get_measurement().measurement_id;

                std::unordered_map<std::size_t, std::size_t>::iterator meas_it =
                    initial_measurement_count.find(meas_id);

                if (meas_it == initial_measurement_count.end()) {
                    // Should never happen
                    verbose_error(
                        "track_index(" + std::to_string(track_index) +
                        ") which is a removed track, has a measurement not "
                        "present in initial_measurement_count. This should "
                        "never happen and is an implementation error.");
                    all_removed_tracks_alright = false;
                } else if (meas_it->second > 1) {
                    ++shared_hits;
                }
            }

            if (shared_hits < _config.maximum_shared_hits) {
                verbose_error(
                    "track_index(" + std::to_string(track_index) +
                    ") which is a removed track, should at least share " +
                    std::to_string(_config.maximum_shared_hits) +
                    " measurement(s) with other tracks, but only shares " +
                    std::to_string(shared_hits) + ".");
                all_removed_tracks_alright = false;
            }
        }
    }

    if (all_removed_tracks_alright) {
        verbose_info(
            "OK 1/2: every removed track had at least one commun measurement "
            "with another track.");
    }

    // ===========================================================================
    // Checks that returned tracks are independent of each other: they should
    // share a maximum of (_config.maximum_shared_hits - 1) hits per track.
    // ===========================================================================

    // Used for return value and final message
    bool independent_tracks = true;

    // Associates each measurement_id to a list of (track_index)es
    std::unordered_map<std::size_t, std::vector<std::size_t>>
        tracks_per_measurements;

    // Only for measurements shared between too many tracks: associates each
    // measurement_id to a list of (track_index)es
    std::unordered_map<std::size_t, std::vector<std::size_t>>
        tracks_per_meas_err;

    // Initializes tracks_per_measurements
    for (std::size_t track_index : final_state.selected_tracks) {
        auto const& [fit_res, states] = initial_track_states.at(track_index);
        for (auto const& mes : states) {
            std::size_t meas_id = mes.get_measurement().measurement_id;
            tracks_per_measurements[meas_id].push_back(track_index);
        }
    }

    // Displays common tracks per measurement if it exceeds the maximum count
    for (auto const& val : tracks_per_measurements) {
        auto const& tracks_per_mes = val.second;
        if (tracks_per_mes.size() > _config.maximum_shared_hits) {
            std::string msg =
                "Measurement " + std::to_string(val.first) +
                " is shared between " + std::to_string(tracks_per_mes.size()) +
                " tracks, superior to _config.maximum_shared_hits(" +
                std::to_string(_config.maximum_shared_hits) +
                "). It is shared between tracks:";
            for (std::size_t track_index : tracks_per_mes) {
                msg += " " + std::to_string(track_index);
            }
            verbose_error(msg);

            // Displays each track's measurements:
            for (std::size_t track_index : tracks_per_mes) {
                msg = "    Track(" + std::to_string(track_index) +
                      ")'s measurements:";
                auto const& [fit_res, states] =
                    initial_track_states.at(track_index);
                for (auto const& st : states) {
                    auto meas_id = st.get_measurement().measurement_id;
                    msg += " " + std::to_string(meas_id);
                }
                verbose_error(msg);
            }

            independent_tracks = false;
        }
    }

    if (independent_tracks) {
        verbose_info(
            "OK 2/2: each selected_track shares at most "
            "(_config.maximum_shared_hits - 1)(=" +
            std::to_string(_config.maximum_shared_hits - 1) +
            ") measurement(s)");
    }

    return (all_removed_tracks_alright && independent_tracks);
}

namespace {

/// Removes a track from the state which has to be done for multiple properties
/// because of redundancy.
static void remove_track(greedy_ambiguity_resolution_algorithm::state_t& state,
                         std::size_t track_index) {
    for (auto meas_index : state.measurements_per_track[track_index]) {
        state.tracks_per_measurement[meas_index].erase(track_index);

        if (state.tracks_per_measurement[meas_index].size() == 1) {
            auto j_track = *state.tracks_per_measurement[meas_index].begin();
            --state.shared_measurements_per_track[j_track];
        }
    }
    state.selected_tracks.erase(track_index);
}
}  // namespace

void greedy_ambiguity_resolution_algorithm::resolve(state_t& state) const {
    /// Compares two tracks based on the number of shared measurements in order
    /// to decide if we already met the final state.
    auto shared_measurements_comperator = [&state](std::size_t a,
                                                   std::size_t b) {
        return state.shared_measurements_per_track[a] <
               state.shared_measurements_per_track[b];
    };

    /// Compares two tracks in order to find the one which should be evicted.
    /// First we compare the relative amount of shared measurements. If that is
    /// indecisive we use the chi2.
    auto track_comperator = [&state](std::size_t a, std::size_t b) {
        /// Helper to calculate the relative amount of shared measurements.
        auto relative_shared_measurements = [&state](std::size_t i) {
            return 1.0 * state.shared_measurements_per_track[i] /
                   state.measurements_per_track[i].size();
        };

        if (relative_shared_measurements(a) !=
            relative_shared_measurements(b)) {
            return relative_shared_measurements(a) <
                   relative_shared_measurements(b);
        }
        return state.track_chi2[a] < state.track_chi2[b];
    };

    std::size_t iteration_count = 0;
    for (std::size_t i = 0; i < _config.maximum_iterations; ++i) {
        // Lazy out if there is nothing to filter on.
        if (state.selected_tracks.empty()) {
            verbose_info("No tracks left - exit loop");
            break;
        }

        // Find the maximum amount of shared measurements per track to decide if
        // we are done or not.
        auto maximum_shared_measurements = *std::max_element(
            state.selected_tracks.begin(), state.selected_tracks.end(),
            shared_measurements_comperator);

        verbose_flood("Current maximum shared measurements " +
                      std::to_string(state.shared_measurements_per_track
                                         [maximum_shared_measurements]));

        if (state.shared_measurements_per_track[maximum_shared_measurements] <
            _config.maximum_shared_hits) {
            break;
        }

        // Find the "worst" track by comparing them to each other
        auto bad_track =
            *std::max_element(state.selected_tracks.begin(),
                              state.selected_tracks.end(), track_comperator);

        verbose_flood(
            "Remove track " + std::to_string(bad_track) + " n_meas " +
            std::to_string(state.measurements_per_track[bad_track].size()) +
            " nShared " +
            std::to_string(state.shared_measurements_per_track[bad_track]) +
            " chi2 " + std::to_string(state.track_chi2[bad_track]));
        remove_track(state, bad_track);
        ++iteration_count;
    }

    verbose_info("Iteration_count: " + std::to_string(iteration_count));
}

}  // namespace traccc
