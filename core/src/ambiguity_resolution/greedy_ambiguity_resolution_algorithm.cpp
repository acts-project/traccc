/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/track_candidate.hpp"

// System include
#include <algorithm>
#include <vector>

namespace traccc {

enum class resolution_status : uint32_t { UNKNOWN, REJECT, ACCEPT, MAX_STATUS };

track_candidate_container_types::host
greedy_ambiguity_resolution_algorithm::operator()(
    const typename track_candidate_container_types::host& track_candidates)
    const {

    // Make sure that min_shared_meas_for_competition is largen than zero
    assert(m_config.min_shared_meas_for_competition > 0u);

    const std::size_t n_tracks = track_candidates.size();

    // Boolean for acceptance
    std::vector<resolution_status> status(n_tracks, resolution_status::UNKNOWN);

    // Measurement ID and chi2 vector
    std::vector<std::vector<std::size_t>> meas_ids(n_tracks);
    std::vector<traccc::scalar> chi_squares(n_tracks);

    // Fill the measurement id vector
    for (std::size_t i = 0; i < n_tracks; i++) {

        // Fill chi square
        chi_squares.at(i) = track_candidates.at(i).header.trk_quality.chi2;

        // Fill measurement Ids
        const auto& candidates = track_candidates.at(i).items;
        meas_ids.at(i).reserve(candidates.size());
        for (const auto& cand : candidates) {
            meas_ids.at(i).push_back(cand.measurement_id);
        }

        // Need to sort the Ids to use set_intersection later
        std::sort(meas_ids.at(i).begin(), meas_ids.at(i).end());

        // Reject if the number of measurements is less the cut
        if (meas_ids.at(i).size() < m_config.min_meas_per_track) {
            status[i] = resolution_status::REJECT;
        }
    }

    // Count the number of shared measurements
    std::vector<std::size_t> n_shared(n_tracks);

    for (std::size_t i = 0; i < n_tracks; i++) {

        std::vector<std::size_t> shared;

        for (std::size_t j = 0; j < n_tracks; j++) {
            if (i != j) {
                std::set_intersection(
                    meas_ids.at(i).begin(), meas_ids.at(i).end(),
                    meas_ids.at(j).begin(), meas_ids.at(j).end(),
                    std::back_inserter(shared));

                // Remove common ids so that 'shared' vector has only unique
                // ids
                std::sort(shared.begin(), shared.end());
                shared.erase(std::unique(shared.begin(), shared.end()),
                             shared.end());
            }
        }

        n_shared.at(i) = shared.size();
        if ((n_shared.at(i) < m_config.min_shared_meas_for_competition) &&
            (status[i] == resolution_status::UNKNOWN)) {
            status[i] = resolution_status::ACCEPT;
        }
    }

    // Iterate over tracks with unknown status
    for (std::size_t i = 0; i < n_tracks; i++) {

        if (status.at(i) == resolution_status::UNKNOWN) {
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
                        auto rel_shared_i =
                            static_cast<traccc::scalar>(n_shared.at(i)) /
                            static_cast<traccc::scalar>(meas_ids.at(i).size());
                        auto rel_shared_j =
                            static_cast<traccc::scalar>(n_shared.at(j)) /
                            static_cast<traccc::scalar>(meas_ids.at(j).size());

                        if (rel_shared_i > rel_shared_j) {
                            status.at(i) = resolution_status::REJECT;
                            break;
                        } else if (rel_shared_i == rel_shared_j) {
                            if (chi_squares.at(i) > chi_squares.at(j)) {
                                status.at(i) = resolution_status::REJECT;
                                break;
                            }
                        }
                    }
                }
            }

            // If the status is still UNKNOWN, tag it as ACCEPT
            if (status.at(i) == resolution_status::UNKNOWN) {
                status.at(i) = resolution_status::ACCEPT;
            }
        }
    }

    // Copy the tracks to be retained in the return value
    track_candidate_container_types::host output;
    output.reserve(
        std::count(status.begin(), status.end(), resolution_status::ACCEPT));

    for (std::size_t i = 0; i < n_tracks; i++) {
        if (status.at(i) == resolution_status::ACCEPT) {
            output.push_back(std::move(track_candidates.at(i).header),
                             std::move(track_candidates.at(i).items));
        }
    }

    return output;
}

}  // namespace traccc
