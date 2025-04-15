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

namespace traccc::host {

track_candidate_container_types::host
greedy_ambiguity_resolution_algorithm::operator()(
    const track_candidate_container_types::const_view& track_candidates_view)
    const {

    const track_candidate_container_types::const_device track_candidates(
        track_candidates_view);

    const std::size_t n_tracks = track_candidates.size();

    // Make the output container
    track_candidate_container_types::host output{&m_mr.main};

    if (n_tracks == 0) {
        return output;
    }

    // Make sure that max_shared_meas is largen than zero
    assert(m_config.max_shared_meas > 0u);

    // Accepted ids to iterate
    std::vector<unsigned int> accepted_ids(n_tracks);
    std::iota(accepted_ids.begin(), accepted_ids.end(), 0);

    // Make measurement ID, pval and n_measurement vector
    std::vector<std::vector<std::size_t>> meas_ids(n_tracks);
    std::vector<traccc::scalar> pvals(n_tracks);
    std::vector<std::size_t> n_meas(n_tracks);

    for (unsigned int i = 0; i < n_tracks; i++) {
        // Fill the pval vectors
        pvals[i] = track_candidates.at(i).header.trk_quality.pval;

        const auto& candidates = track_candidates.at(i).items;
        const unsigned int n_cands = candidates.size();

        if (n_cands < m_config.min_meas_per_track) {
            // Reject if the number of measurements is less than the cut
            const auto it =
                std::lower_bound(accepted_ids.begin(), accepted_ids.end(), i);
            assert(it != accepted_ids.end() && *it == i);
            accepted_ids.erase(it);
        } else {
            // Fill measurement ids and n_measurements
            meas_ids[i].reserve(n_cands);
            for (const auto& cand : candidates) {
                meas_ids[i].push_back(cand.measurement_id);
            }
            n_meas[i] = n_cands;
            assert(n_cands == meas_ids[i].size());
        }
    }

    // Record the tracks per measurement
    std::vector<std::vector<unsigned int>> tracks_per_measurement;

    for (const auto& i : accepted_ids) {
        const auto& candidates = track_candidates.at(i).items;
        for (const auto& cand : candidates) {
            if (tracks_per_measurement.size() < cand.measurement_id + 1) {
                tracks_per_measurement.resize(cand.measurement_id + 1);
            }
            tracks_per_measurement[cand.measurement_id].push_back(i);
        }
    }

    // Count the number of shared measurements
    std::vector<unsigned int> n_shared(n_tracks);
    for (const auto& i : accepted_ids) {
        for (const auto& meas_id : meas_ids[i]) {
            if (tracks_per_measurement[meas_id].size() > 1) {
                n_shared[i]++;
            }
        }
    }

    // Precompute max sizes for 2D cache table
    const auto max_n_meas = *std::max_element(n_meas.begin(), n_meas.end());
    const auto max_n_shared =
        *std::max_element(n_shared.begin(), n_shared.end());

    // Initialize 2D cache table for relative shared measurements
    std::vector<std::vector<traccc::scalar>> cache(
        max_n_meas + 1, std::vector<traccc::scalar>(max_n_shared + 1));
    std::vector<traccc::scalar> rel_shared(n_tracks);
    for (const auto& i : accepted_ids) {
        const auto n_s = n_shared[i];
        const auto n_m = n_meas[i];
        assert(n_m <= max_n_meas && n_s <= max_n_shared);
        if (cache[n_m][n_s] == 0.f && n_s > 0) {
            cache[n_m][n_s] = static_cast<traccc::scalar>(n_s) /
                              static_cast<traccc::scalar>(n_m);
        }
        rel_shared[i] = cache[n_m][n_s];
    }

    // Sort the track id with rel_shared and pval to find the worst track fast
    std::vector<unsigned int> sorted_ids = accepted_ids;

    auto track_comparator = [&rel_shared, &pvals](unsigned int a,
                                                  unsigned int b) {
        if (rel_shared[a] != rel_shared[b]) {
            return rel_shared[a] < rel_shared[b];
        }
        return pvals[a] > pvals[b];
    };
    std::sort(sorted_ids.begin(), sorted_ids.end(), track_comparator);

    // Iterate over tracks
    for (unsigned int iter = 0; iter < m_config.max_iterations; iter++) {
        // Terminate if there are no tracks to iterate
        if (accepted_ids.empty()) {
            break;
        }

        unsigned int max_shared{0u};
        for (const auto& i : accepted_ids) {
            if (n_shared[i] > max_shared)
                max_shared = n_shared[i];
        }

        // Terminate if the max shared measurements is less than the cut value
        if (max_shared < m_config.max_shared_meas) {
            break;
        }

        // The last element of sorted vector is the worst track
        const unsigned int worst_track = sorted_ids.back();

        // Remove the worst track from the accepted ids
        const auto it1 = std::lower_bound(accepted_ids.begin(),
                                          accepted_ids.end(), worst_track);
        assert(it1 != accepted_ids.end() && *it1 == worst_track);
        accepted_ids.erase(it1);

        // Pop the worst (rejected) id from the sorted ids
        sorted_ids.pop_back();

        const auto& meas_ids_to_remove = meas_ids[worst_track];
        for (const auto& id : meas_ids_to_remove) {
            auto& tracks = tracks_per_measurement[id];

            // Remove the worst (rejected) id from the tracks associated with
            // measurement
            const auto it2 =
                std::lower_bound(tracks.begin(), tracks.end(), worst_track);
            assert(it2 != tracks.end() && *it2 == worst_track);
            tracks.erase(it2);

            // If there is only one track associated with measurement, the
            // number of shared measurement can be reduced by one
            if (tracks.size() == 1) {
                const auto tid = tracks[0];
                n_shared[tid]--;

                const auto n_s = n_shared[tid];
                const auto n_m = n_meas[tid];
                assert(n_m <= max_n_meas && n_s <= max_n_shared);
                if (cache[n_m][n_s] == 0.f && n_s > 0) {
                    cache[n_m][n_s] = static_cast<traccc::scalar>(n_s) /
                                      static_cast<traccc::scalar>(n_m);
                }
                rel_shared[tid] = cache[n_m][n_s];

                // Reposition the track to the next of the worse track which is
                // firstly found during the reverse iteration
                const auto it3 =
                    std::find(sorted_ids.begin(), sorted_ids.end(), tid);
                assert(it3 != sorted_ids.end());
                const auto it4 = std::lower_bound(sorted_ids.begin(), it3, tid,
                                                  track_comparator);
                sorted_ids.erase(it3);
                sorted_ids.insert(it4, tid);
            }
        }

        // Make sure that sorted_ids stays sorted
        assert(std::is_sorted(sorted_ids.begin(), sorted_ids.end(),
                              track_comparator));
    }

    // Fill the output container with accepted tracks
    output.reserve(accepted_ids.size());
    for (const auto& i : accepted_ids) {
        vecmem::vector<traccc::track_candidate> output_cands;

        const auto& input_cands = track_candidates.at(i).items;

        output_cands.insert(output_cands.end(), input_cands.begin(),
                            input_cands.end());

        output.push_back(std::move(track_candidates.at(i).header),
                         std::move(output_cands));
    }

    return output;
}

}  // namespace traccc::host
