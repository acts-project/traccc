/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "traccc/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"

// System include
#include <algorithm>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

// VecMem include(s).
#include <vecmem/containers/vector.hpp>

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/utils/algorithm.hpp"

// Greedy ambiguity resolution adapted from ACTS code

namespace traccc {

/// Run the algorithm
///
/// @param track_states the container of the fitted track parameters
/// @return the container without ambiguous tracks
track_state_container_types::host
greedy_ambiguity_resolution_algorithm::operator()(
    const typename track_state_container_types::host& track_states) const {

    // Boolean for acceptance
    std::vector<bool> acceptance(n_tracks, true);

    // Measurement id vector
    std::vector<std::vector<std::size_t>> meas_ids(n_track);

    // Fill the measurement id vector
    for (std::size_t i = 0; i < n_tracks; i++) {
        const auto& states = track_states.at(i_trk).items;
        meas_ids.at(i).reserve(states.size());
        for (const auto& st : states) {
            meas_ids.at(i).push_back(st.get_measurement().measurement_id);
        }
        // Need to sort to use set_intersection later
        std::sort(meas_ids.at(i).begin(), meas_ids.at(i).end());
    }

    // Number of shared measurements per track
    std::vector<std::size_t> n_shared_meas_per_track;
    n_shared_meas_per_track.reserve(n_tracks);

    // Count the number of shared measurements
    for (std::size_t i = 0; i < n_tracks; i++) {
        const auto& ref_ids = meas_ids.at(i);
        std::vector<std::size_t> shared_ids;

        for (std::size_t j = 0; j < n_tracks; j++) {
            const auto& comp_ids = meas_ids.at(j);
            std::set_intersection(ref_ids.begin(), ref_ids.end(),
                                  comp_ids.begin(), comp_ids.end(),
                                  std::back_inserter(shared_ids));
        }
    }

    for (std::size_t i_trk = 0; i_trk < n_tracks; i_trk++) {
        auto const& [fit_res, states] = track_states.at(i_trk);

        // Kick out tracks that do not fulfill our initial requirements
        if (states.size() < m_config.min_measurements_per_track) {
            acceptance[i_trk] = false;
        }
    }

    const std::size_t n_tracks = track_states.size();

    // Vector of pairs of < measurement ID, number of occurences in patterns >
    // vecmem::vector<traccc::pair<std::size_t, unsigned int>>
    // meas_id_occurence;

    // Vector of unique measurement ids
    vecmem::vector<std::size_t> meas_ids;
    // Vector of vector of track ids associated with each measurement id
    vecmem::jagged_vector<std::size_t> track_ids;

    // Copy the tracks to be retained in the return value
    track_state_container_types::host output;
    output.reserve(state.selected_tracks.size());

    for (std::size_t index : state.selected_tracks) {
        output.push_back(std::move(track_states.at(index).header),
                         std::move(track_states.at(index).items));
    }

    return output;
}

void greedy_ambiguity_resolution_algorithm::compute_initial_state(
    const typename track_state_container_types::host& track_states,
    state_t& state) const {

    // @TODO: Add assertion to check the number of mcount_id_zero

    // For each track of the input container
    const std::size_t n_track_states = track_states.size();
    for (std::size_t track_index = 0; track_index < n_track_states;
         ++track_index) {

        auto const& [fit_res, states] = track_states.at(track_index);

        // Kick out tracks that do not fulfill our initial requirements
        if (states.size() < m_config.n_measurements_min) {
            continue;
        }

        // Create the list of measurement_id of the current track
        std::vector<std::size_t> measurements;
        std::unordered_map<std::size_t, std::size_t> already_added_mes;

        for (auto const& st : states) {
            std::size_t mid = st.get_measurement().measurement_id;

            // If the same measurement is found multiple times in a single
            // track: remove duplicates.
            auto dm = already_added_mes.insert(
                std::pair<std::size_t, std::size_t>(mid, 1));

            // If the measurement was already present in already_added_mes
            if (!dm.second) {
                // Increment the count for this measurement_id
                ++(dm.first->second);
                TRACCC_DEBUG("(1/3) Track " << track_index
                                            << " has duplicated measurement "
                                            << mid << ".");
            } else {
                measurements.push_back(mid);
            }
        }

        // Add this track chi2 value
        state.track_chi2.push_back(fit_res.trk_quality.chi2);
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
/// - Returned tracks should be independent of each other: they should
/// share a maximum of (_config.maximum_shared_hits - 1) hits per track.
/// - Each removed track should share at least
/// (_config.maximum_shared_hits) with another initial track.
///
/// @param initial_track_states The input track container, as given to
/// compute_initial_state.
/// @param final_state The state object after the resolve method has
/// been called.
bool greedy_ambiguity_resolution_algorithm::check_obvious_errors(
    const typename track_state_container_types::host& initial_track_states,
    state_t& final_state) const {

    // Associates every measurement_id to the number of tracks that shares it
    // (during initial state)
    std::unordered_map<std::size_t, std::size_t> initial_measurement_count;

    // Initialize initial_measurement_count
    for (std::size_t track_index = 0; track_index < initial_track_states.size();
         ++track_index) {
        // fit_res is a fitting_result<default_algebra>
        // states  is a vecmem_vector<track_state<default_algebra>>
        auto const& [fit_res, states] = initial_track_states.at(track_index);

        std::set<std::size_t> already_added_mes;

        for (auto const& st : states) {
            std::size_t meas_id = st.get_measurement().measurement_id;

            // If the same measurement is found multiple times in a single
            // track: remove duplicates.
            if (already_added_mes.find(meas_id) != already_added_mes.end()) {
                TRACCC_DEBUG("(2/3) Track " << track_index
                                            << " has duplicated measurement "
                                            << meas_id << ".");
                continue;
            }
            already_added_mes.insert(meas_id);

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
    // =========================================================================
    // Checks that every removed track had at least
    // (_config.maximum_shared_hits) common measurements with other tracks
    // =========================================================================
    std::size_t n_initial_track_states = initial_track_states.size();
    for (std::size_t track_index = 0; track_index < n_initial_track_states;
         ++track_index) {
        auto const& [fit_res, states] = initial_track_states.at(track_index);

        // Skip this track if it has to be kept (i.e. exists in selected_tracks)
        if (final_state.selected_tracks.find(track_index) !=
            final_state.selected_tracks.end()) {
            continue;
        }

        // So if the current track has been removed from selected_tracks:

        std::size_t shared_hits = 0;
        for (auto const& st : states) {
            auto meas_id = st.get_measurement().measurement_id;

            std::unordered_map<std::size_t, std::size_t>::iterator meas_it =
                initial_measurement_count.find(meas_id);

            if (meas_it == initial_measurement_count.end()) {
                // Should never happen
                TRACCC_ERROR(
                    "track_index("
                    << track_index
                    << ") which is a removed track, has a measurement not "
                    << "present in initial_measurement_count. This should "
                    << "never happen and is an implementation error.\n");
                all_removed_tracks_alright = false;
            } else if (meas_it->second > 1) {
                ++shared_hits;
            }
        }

        if (shared_hits < _config.maximum_shared_hits) {
            TRACCC_ERROR(
                "track_index("
                << track_index
                << ") which is a removed track, should at least share "
                << _config.maximum_shared_hits
                << " measurement(s) with other tracks, but only shares "
                << shared_hits);
            all_removed_tracks_alright = false;
        }
    }

    if (all_removed_tracks_alright) {
        TRACCC_INFO(
            "OK 1/2: every removed track had at least one common measurement "
            "with another track.");
    }

    // =========================================================================
    // Checks that returned tracks are independent of each other: they should
    // share a maximum of (_config.maximum_shared_hits - 1) hits per track.
    // =========================================================================

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

        std::set<std::size_t> already_added_mes;

        for (auto const& mes : states) {
            std::size_t meas_id = mes.get_measurement().measurement_id;

            // If the same measurement is found multiple times in a single
            // track: remove duplicates.
            if (already_added_mes.find(meas_id) != already_added_mes.end()) {
                TRACCC_DEBUG("(3/3) Track " << track_index
                                            << " has duplicated measurement "
                                            << meas_id << ".");
            } else {
                already_added_mes.insert(meas_id);
                tracks_per_measurements[meas_id].push_back(track_index);
            }
        }
    }

    // Displays common tracks per measurement if it exceeds the maximum count
    for (auto const& val : tracks_per_measurements) {
        auto const& tracks_per_mes = val.second;
        if (tracks_per_mes.size() > _config.maximum_shared_hits) {
            std::stringstream ss;
            ss << "Measurement " << val.first << " is shared between "
               << tracks_per_mes.size()
               << " tracks, superior to _config.maximum_shared_hits("
               << _config.maximum_shared_hits
               << "). It is shared between tracks:";

            for (std::size_t track_index : tracks_per_mes) {
                ss << " " << track_index;
            }

            TRACCC_ERROR(ss.str());

            // Displays each track's measurements:
            for (std::size_t track_index : tracks_per_mes) {
                std::stringstream ssm;
                ssm << "    Track(" << track_index << ")'s measurements:";
                auto const& [fit_res, states] =
                    initial_track_states.at(track_index);

                for (auto const& st : states) {
                    auto meas_id = st.get_measurement().measurement_id;
                    ssm << " " << meas_id;
                }
                TRACCC_ERROR(ssm.str());
            }

            independent_tracks = false;
        }
    }

    if (independent_tracks) {
        TRACCC_INFO(
            "OK 2/2: each selected_track shares at most "
            "(_config.maximum_shared_hits - 1)(="
            << _config.maximum_shared_hits - 1 << ") measurement(s)");
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
            return static_cast<double>(state.shared_measurements_per_track[i]) /
                   static_cast<double>(state.measurements_per_track[i].size());
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
            TRACCC_DEBUG("No tracks left - exit loop");
            break;
        }

        // Find the maximum amount of shared measurements per track to decide if
        // we are done or not.
        auto maximum_shared_measurements = *std::max_element(
            state.selected_tracks.begin(), state.selected_tracks.end(),
            shared_measurements_comperator);

        TRACCC_DEBUG(
            "Current maximum shared measurements "
            << state
                   .shared_measurements_per_track[maximum_shared_measurements]);

        if (state.shared_measurements_per_track[maximum_shared_measurements] <
            _config.maximum_shared_hits) {
            break;
        }

        // Find the "worst" track by comparing them to each other
        auto bad_track =
            *std::max_element(state.selected_tracks.begin(),
                              state.selected_tracks.end(), track_comperator);

        TRACCC_DEBUG("Remove track "
                     << bad_track << " n_meas "
                     << state.measurements_per_track[bad_track].size()
                     << " nShared "
                     << state.shared_measurements_per_track[bad_track]
                     << " chi2 " << state.track_chi2[bad_track]);

        remove_track(state, bad_track);
        ++iteration_count;
    }

    TRACCC_DEBUG("Iteration_count: " << iteration_count);
}

}  // namespace traccc
