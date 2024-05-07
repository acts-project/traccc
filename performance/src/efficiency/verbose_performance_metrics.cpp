/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/utils/helpers.hpp"

// Project include(s).
// #include "duplication_plot_tool.hpp"
// #include "eff_plot_tool.hpp"
// #include "fake_tracks_plot_tool.hpp"
// #include "traccc/edm/track_candidate.hpp"
// #include "traccc/edm/track_state.hpp"
#include <traccc/efficiency/verbose_performance_metrics.hpp>

#include "traccc/io/event_map2.hpp"
#include "track_classification.hpp"

// System include(s).
#include <map>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <vector>

// All in the header file, then will be split.

namespace traccc {
namespace details {

namespace {

/**
 * @brief For track finding only. Associates each reconstructed track with its
 * measurements.
 *
 * @param track_candidates_view the track candidates found by the finding
 * algorithm.
 * @return std::vector<std::vector<measurement>> Associates each track index
 * with its corresponding measurements.
 */
std::vector<std::vector<measurement>> prepare_data(
    const track_candidate_container_types::const_view& track_candidates_view) {
    std::vector<std::vector<measurement>> result;

    // Iterate over the tracks.
    track_candidate_container_types::const_device track_candidates(
        track_candidates_view);

    const unsigned int n_tracks = track_candidates.size();
    result.reserve(n_tracks);

    for (unsigned int i = 0; i < n_tracks; i++) {
        const auto& cands = track_candidates.at(i).items;

        std::vector<measurement> measurements;
        measurements.reserve(cands.size());
        for (const auto& cand : cands) {
            measurements.push_back(cand);
        }
        result.push_back(std::move(measurements));
    }
    return result;
}

/**
 * @brief For ambiguity resolution and track fitting only. Associates each
 * reconstructed track with its measurements.
 *
 * @param track_candidates_view the track candidates found by the finding
 * algorithm.
 * @return std::vector<std::vector<measurement>> Associates each track index
 * with its corresponding measurements.
 */
std::vector<std::vector<measurement>> prepare_data(
    const track_state_container_types::const_view& track_states_view) {
    std::vector<std::vector<measurement>> result;

    // Iterate over the tracks.
    track_state_container_types::const_device track_states(track_states_view);

    const unsigned int n_tracks = track_states.size();
    result.reserve(n_tracks);

    for (unsigned int i = 0; i < n_tracks; i++) {
        auto const& [fit_res, states] = track_states.at(i);
        std::vector<measurement> measurements;
        measurements.reserve(states.size());
        for (const auto& st : states) {
            measurements.push_back(st.get_measurement());
        }
        result.push_back(std::move(measurements));
    }
    return result;
}
}  // namespace

void verbose_performance_metrics::ambiguity_resolution(
    std::vector<std::vector<measurement>> const& all_tracks,
    std::vector<std::size_t> const& selected_tracks,
    const event_map2& evt_map) {

    // TODO: error message
    if (all_tracks.size() == 0)
        return;

    // TODO: error message
    if (selected_tracks.size() == 0)
        return;

    // In the following comments, "particle" means "truth particle"
    // and "track" means "reconstructed track" (by the ambiguity resolution
    // algorithm)

    // For the tracks solely made from the hits of one single particle.
    // TODO: check that the vector size always equals 1.
    // <particle_id, std::vector<track_id>>
    // single_tp_selected_tracks;
    std::map<std::size_t, std::vector<std::size_t>> single_tp_selected_tracks;

    // for storing all the tracks (evicted and selected) that are solely
    // made from the hits of a single truth particle.
    //<particle_id, std::vector<reconstructed_track_id>>
    std::map<std::size_t, std::vector<std::size_t>> single_tp_all_tracks;

    std::size_t selected_index_sum = 0;
    std::size_t selected_fake = 0;
    std::size_t evicted_fake = 0;
    std::size_t evicted_duplicates = 0;
    // std::size_t evicted_single_tp_count = 0;

    // For each track_index, true if it's made entirely of only one truth
    // particle
    std::vector<bool> track_single_tp;
    track_single_tp.resize(all_tracks.size());

    // For each track_index:
    //    if (track_single_tp[track_index]) the id of the truth particle
    //    otherwise undefined.
    std::vector<std::size_t> track_single_tp_pid;
    track_single_tp_pid.resize(all_tracks.size());

    // True if the track has been selected, false if evicted
    std::vector<bool> track_is_selected(all_tracks.size(), false);

    // Iterate over all tracks
    for (std::size_t track_index = 0; track_index < all_tracks.size();
         ++track_index) {

        std::vector<measurement> const& measurements = all_tracks[track_index];

        std::vector<particle_hit_count> particle_hit_counts =
            identify_contributing_particles(measurements, evt_map.meas_ptc_map);
        bool single_tp = (particle_hit_counts.size() == 1);
        track_single_tp[track_index] = single_tp;

        // If the track is solely made from the hits of a single truth particle,
        // add the track_index to single_tp_all_tracks. It will be used to
        // determine the score for each selected track (the track having the
        // most measurements).
        if (single_tp) {
            std::size_t particle_id = particle_hit_counts[0].ptc.particle_id;
            std::vector<std::size_t>& tp_tracks =
                single_tp_all_tracks[particle_id];
            tp_tracks.push_back(track_index);
            track_single_tp_pid[track_index] = particle_id;
        }
        // If the track shares hits from different truth particles, ignore
        // the track.
        // if (particle_hit_counts.size() > 1) {}
    }

    // Iterate over all single_tp_all_tracks entries
    for (auto it = single_tp_all_tracks.begin();
         it != single_tp_all_tracks.end(); ++it) {
        std::vector<std::size_t>& tp_tracks = it->second;
        // Sort the std::vector<reconstructed_track_index> according to the
        // number of hits of each track, descending. We now have, for each
        // truth particle, a ranking from the “best” reconstructed track to
        // the “worst” reconstructed track representing the truth particle.
        // It will be used to have a number to understand how well the
        // ambiguity solver chose each selected track.
        auto order_tracks = [&](std::size_t ti_a, std::size_t ti_b) {
            std::size_t hc_a = all_tracks[ti_a].size();  // hit count for a
            std::size_t hc_b = all_tracks[ti_b].size();  // hit count for b
            // Sort descending, greatest number of measurements first.
            return hc_a > hc_b;
        };
        std::sort(tp_tracks.begin(), tp_tracks.end(), order_tracks);
    }

    // Iterate over all selected tracks
    for (std::size_t track_index : selected_tracks) {
        track_is_selected[track_index] = true;

        // If the track is solely made from the hits of a single truth
        // particle, add it to single_tp_selected_tracks, and find the
        // corresponding entry in single_tp_all_tracks: find the particle_id
        // in the map, and the index of the track. selected_index_sum +=
        // (its index in the vector), but skipping tracks that have the same
        // number of measurements. Two tracks with the same number of
        // measurements must have the same “score”. Also increment the
        // selected_count.
        if (track_single_tp[track_index]) {

            std::size_t pid = track_single_tp_pid[track_index];
            std::vector<std::size_t>& tp_tracks =
                single_tp_selected_tracks[pid];
            std::vector<std::size_t>& tp_common_tracks =
                single_tp_all_tracks[pid];

            // TODO: a vector should be useless, a single value should
            // suffice. TODO: check that no entry already existed in
            // single_tp_selected_tracks.
            tp_tracks.push_back(track_index);

            // The number of hits of the first sorted track
            std::size_t hits;
            // Current track score (lower is better)
            std::size_t score = 0;
            bool debug_found = false;

            // For each track made solely of the truth particle
            for (std::size_t i = 0; i < tp_common_tracks.size(); ++i) {
                std::size_t ti = tp_common_tracks[i];  // track index

                // Track found
                if (ti == track_index) {
                    debug_found = true;
                    selected_index_sum += score;
                    break;
                }

                // First track and not track_index, initialize hits
                if (i == 0) {
                    hits = all_tracks[ti].size();
                } else {
                    std::size_t new_hits = all_tracks[ti].size();
                    // equivalent to new_hits < hits
                    if (new_hits != hits) {
                        ++score;  // worsen the score
                    }
                    if (new_hits > hits) {
                        // TODO print error message
                    }
                    hits = new_hits;
                }
            }

            if (!debug_found) {
                // TODO print error message
            }
        }
        // If the track shares hits from multiple truth particles: increment
        // the selected_fake counter.
        else {
            ++selected_fake;
        }
    }

    std::set<std::size_t> evicted_unique_particles;
    // Iterate over all evicted tracks
    for (std::size_t track_index = 0; track_index < all_tracks.size();
         ++track_index) {
        if (track_is_selected[track_index])
            continue;

        // If the track is solely made from the hits of a single truth
        // particle
        if (track_single_tp[track_index]) {
            std::size_t pid = track_single_tp_pid[track_index];
            // if the associated truth particle is present in
            // single_tp_selected_tracks, increment the
            // evicted_duplicates (meaning we kept at least one track
            // representing the truth particle)
            auto it = single_tp_selected_tracks.find(pid);
            if (it != single_tp_selected_tracks.end()) {
                ++evicted_duplicates;
            } else {
                // otherwise we have made a mistake and evicted a valid
                // track, without redundancy: add the particle_id to
                // evicted_unique_particles
                evicted_unique_particles.insert(pid);
            }
        } else {
            // If the track shares hits from multiple truth particles:
            // increment the evicted_fake counter
            ++evicted_fake;
        }
    }

    std::size_t selected_duplicates = 0;
    for (auto const& it : single_tp_selected_tracks) {
        // std::size_t pid = it.first;
        selected_duplicates += static_cast<int>(it.second.size()) - 1;
        // if (selected_duplicates < 10) {
        // std::cout << "particle_id " << it.first << "   "
        //           << "selected_duplicates " << selected_duplicates << "   "
        //           << ", it.second.size() = " << it.second.size() << "\n";
        // }
    }

    // std::size_t selected_count = selected_tracks.size();
    std::size_t selected_valid =
        selected_tracks.size() - selected_fake - selected_duplicates;

    // Or, to make it also available for finding:
    // number of track duplicates
    // number of fake tracks
    // number of truth particles

    // Among the selected tracks:
    // number of track duplicates (should be zero)
    // number of valid tracks (= number of truth particles minus duplicates)
    // number of fake tracks (that should have been removed)

    // Among the evicted tracks
    // number of fake tracks (that were rightly removed)
    // number of duplicates (one particle corresponding to each track is still
    //    present in the selected tracks list)
    // number of removed valid particles (with no redundancy in selected tracks)

    double valid_quality = static_cast<double>(selected_index_sum) /
                           static_cast<double>(selected_tracks.size());

    // Among the selected tracks
    ar_metrics.selected.valid += selected_valid;
    ar_metrics.selected.duplicate += selected_duplicates;
    ar_metrics.selected.fake += selected_fake;

    // Among the evicted tracks
    ar_metrics.evicted.valid += evicted_unique_particles.size();
    ar_metrics.evicted.duplicate += evicted_duplicates;
    ar_metrics.evicted.fake += evicted_fake;

    ++ar_metrics.call_count;

    // Add to the sum of the results quality for valid tracks
    ar_metrics.valid_quality_sum += valid_quality;

    std::cout << "Ambiguity resolution efficiency metrics:\n";

    std::cout << "  Among the selected tracks:\n";
    std::cout << "  Valid quality: " << valid_quality << "\n";
    std::cout << "          Valid: " << selected_valid << "\n";
    std::cout << "     Duplicates: " << selected_duplicates << "\n";
    std::cout << "          Fakes: " << selected_fake << "\n";

    std::cout << "  Among the evicted tracks:\n";
    std::cout << "          Valid: " << evicted_unique_particles.size() << "\n";
    std::cout << "     Duplicates: " << evicted_duplicates << "\n";
    std::cout << "          Fakes: " << evicted_fake << "\n";
}

std::ostream& verbose_performance_metrics::print(std::ostream& os) {

    if (is_ambiguity_resolution) {

        os << "===== Ambiguity resolution performance metrics";
        if (algorithm_name != "")
            os << " (" << algorithm_name << ")";
        os << " =====\n";

        if (ar_metrics.call_count == 0) {
            os << "ERROR: the "
                  "verbose_performance_metrics::ambiguity_resolution "
                  "was never called.";
            return os;
        }

        ar_metrics.selected.compute_percentages();
        ar_metrics.evicted.compute_percentages();
        os << "--Among the selected tracks:\n";
        double quality = ar_metrics.valid_quality_sum / ar_metrics.call_count;
        os << "  Valid quality: " << quality
           << " (should be as low as possible)\n"
           << "          Valid: " << ar_metrics.selected.valid << " ("
           << ar_metrics.selected.valid_p << "%)\n"
           << "     Duplicates: " << ar_metrics.selected.duplicate << " ("
           << ar_metrics.selected.duplicate_p << "%)\n"
           << "          Fakes: " << ar_metrics.selected.fake << " ("
           << ar_metrics.selected.fake_p << "%)\n"
           << "--Among the evicted tracks:\n"
           << "          Valid: " << ar_metrics.evicted.valid << " ("
           << ar_metrics.evicted.valid_p << "%) (not in selected tracks)\n"
           << "     Duplicates: " << ar_metrics.evicted.duplicate << " ("
           << ar_metrics.evicted.duplicate_p << "%)\n"
           << "          Fakes: " << ar_metrics.evicted.fake << " ("
           << ar_metrics.evicted.fake_p << "%)\n";
        return os;

    } else {

        os << "===== Performance metrics for " << algorithm_name << " =====\n";

        gen_metrics.compute_percentages();
        os << "          Valid: " << gen_metrics.valid << " ("
           << gen_metrics.valid_p << "%)\n";
        os << "     Duplicates: " << gen_metrics.duplicate << " ("
           << gen_metrics.duplicate_p << "%)\n";
        os << "          Fakes: " << gen_metrics.fake << " ("
           << gen_metrics.fake_p << "%)\n";

        return os;
    }
}

void verbose_performance_metrics::ambiguity_resolution(
    const track_state_container_types::host& all_tracks,
    const std::vector<std::size_t>& selected_tracks,
    const event_map2& evt_map) {
    std::vector<std::vector<measurement>> all_tracks_vect =
        prepare_data(traccc::get_data(all_tracks));
    ambiguity_resolution(all_tracks_vect, selected_tracks, evt_map);
}

void verbose_performance_metrics::generic(
    const track_candidate_container_types::host& all_tracks,
    const event_map2& evt_map) {
    std::vector<std::vector<measurement>> all_tracks_vect =
        prepare_data(traccc::get_data(all_tracks));
    generic(all_tracks_vect, evt_map);
}

void verbose_performance_metrics::generic(
    const track_state_container_types::host& all_tracks,
    const event_map2& evt_map) {
    std::vector<std::vector<measurement>> all_tracks_vect =
        prepare_data(traccc::get_data(all_tracks));
    generic(all_tracks_vect, evt_map);
}

/// Generic version, for track seeding / finding / fitting
/// But not for ambiguity resolution
void verbose_performance_metrics::generic(
    const std::vector<std::vector<measurement>>& all_tracks,
    const event_map2& evt_map) {

    // TODO: error message
    if (all_tracks.size() == 0)
        return;

    // In the following comments, "particle" means "truth particle"
    // and "track" means "reconstructed track" (by the ambiguity resolution
    // algorithm)

    // For storing all the tracks that are solely made from the hits of a single
    // truth particle.
    //<particle_id, std::vector<reconstructed_track_id>>
    std::map<std::size_t, std::vector<std::size_t>> single_tp_all_tracks;

    std::size_t valid_tracks = 0;
    std::size_t duplicated_tracks = 0;
    std::size_t fake_tracks = 0;

    // For each track_index, true if it's made entirely of only one truth
    // particle
    std::vector<bool> track_single_tp;
    track_single_tp.resize(all_tracks.size());

    // For each track_index:
    //    if (track_single_tp[track_index]) the id of the truth particle
    //    otherwise undefined.
    std::vector<std::size_t> track_single_tp_pid;
    track_single_tp_pid.resize(all_tracks.size());

    // Iterate over all tracks
    for (std::size_t track_index = 0; track_index < all_tracks.size();
         ++track_index) {

        std::vector<measurement> const& measurements = all_tracks[track_index];

        std::vector<particle_hit_count> particle_hit_counts =
            identify_contributing_particles(measurements, evt_map.meas_ptc_map);
        bool single_tp = (particle_hit_counts.size() == 1);
        track_single_tp[track_index] = single_tp;

        // If the track is solely made from the hits of a single truth particle,
        // add the track_index to single_tp_all_tracks. It will be used to
        // determine the score for each selected track (the track having the
        // most measurements).
        if (single_tp) {
            std::size_t particle_id = particle_hit_counts[0].ptc.particle_id;
            std::vector<std::size_t>& tp_tracks =
                single_tp_all_tracks[particle_id];
            tp_tracks.push_back(track_index);
            track_single_tp_pid[track_index] = particle_id;
        } else {
            // If the track shares hits from different truth particles,
            // increment the number of fake tracks.
            ++fake_tracks;
        }
    }

    valid_tracks = single_tp_all_tracks.size();

    // Iterate over tracks made of a single truth particle
    for (const auto& it : single_tp_all_tracks) {
        duplicated_tracks += it.second.size() - 1;
    }

    // Among the selected tracks
    gen_metrics.valid += valid_tracks;
    gen_metrics.duplicate += duplicated_tracks;
    gen_metrics.fake += fake_tracks;
}

}  // namespace details
}  // namespace traccc
