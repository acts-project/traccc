/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/track_state.hpp"

// System include
#include <algorithm>
#include <vector>

namespace traccc {

/// Run the algorithm
///
/// @param track_states the container of the fitted track parameters
/// @return the container without ambiguous tracks
track_state_container_types::host
greedy_ambiguity_resolution_algorithm::operator()(
    const typename track_state_container_types::host& track_states) const {

    const std::size_t n_tracks = track_states.size();

    // Boolean for acceptance
    std::vector<resolution_status> status(n_tracks, resolution_status::UNKNOWN);

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

        // Reject if the number of measurements is less the cut
        if (meas_ids.at(i) < m_config.min_meas_per_track) {
            status[i] = resolution_status::REJECT;
        }
    }

    // Count the number of shared measurements
    std::vector<std::size_t> n_shared(n_track);

    for (std::size_t i = 0; i < n_tracks; i++) {
        std::vector<std::size_t> shared;

        for (std::size_t j = 0; j < n_tracks; j++) {
            if (i != j) {
                std::set_intersection(
                    meas_ids.at(i).begin(), meas_ids.at(i).end(),
                    meas_ids.at(j).begin(), meas_ids.at(j).end(),
                    std::back_inserter(shared));

                // Remove common ids so that 'shared' vector has only unique ids
                std::sort(shared.begin(), shared.end());
                shared.erase(std::unique(shared.begin(), shared.end()),
                             shared.end());
            }
        }

        n_shared.at(i) = shared.size();
        if (n_shared.at(i) < m_config.min_shared_meas_for_competition) {
            status[i] = resolution_status::ACCEPT;
        }
    }

    // Iterate over tracks with unknown status
    for (std::size_t i = 0; i < n_tracks; i++) {
        if (acceptance[i] == resolution_status::UNKNOWN) {
            for (std::size_t j = 0; j < n_tracks; j++) {
                if (i != j) {

                    std::vector<std::size_t> shared;
                    std::set_intersection(
                        meas_ids.at(i).begin(), meas_ids.at(i).end(),
                        meas_ids.at(j).begin(), meas_ids.at(j).end(),
                        std::back_inserter(shared));

                    // If two track have any shared hit and the reference track
                    // is worse, tag it as REJECT
                    if (!shared.empty()) {
                        if (n_shared.at(i) / meas_ids.at(i).size() >
                            n_shared.at(j) / meas_ids.at(j).size()) {
                            acceptance.at(i) = resolution_status::REJECT;
                            break;
                        } else {
                            if (pval.at(i) < pval.at(j)) {
                                acceptance.at(i) = resolution_status::REJECT;
                                break;
                            }
                        }
                    }
                }
            }

            // If the status is still UNKNOWN, tag it as ACCEPT
            if (acceptance[i] == resolution_status::UNKNOWN) {
                acceptance[i] = resolution_status::ACCEPT;
            }
        }
    }

    // Copy the tracks to be retained in the return value
    track_state_container_types::host output;
    output.reserve(std::count(acceptance.begin(), acceptance.end(),
                              resolution_status::ACCEPT));

    for (std::size_t i = 0; i < n_tracks; i++) {
        if (acceptance[i] = resolution_status::ACCEPT) {
            output.push_back(std::move(track_states.at(index).header),
                             std::move(track_states.at(index).items));
        }
    }

    return output;
}

}  // namespace traccc
