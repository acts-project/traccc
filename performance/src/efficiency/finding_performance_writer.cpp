/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/efficiency/finding_performance_writer.hpp"

#include "../resolution/stat_plot_tool.hpp"
#include "duplication_plot_tool.hpp"
#include "eff_plot_tool.hpp"
#include "fake_tracks_plot_tool.hpp"
#include "track_classification.hpp"

// ROOT include(s).
#ifdef TRACCC_HAVE_ROOT
#include <TFile.h>
#endif  // TRACCC_HAVE_ROOT

// System include(s).
#include <iostream>
#include <memory>
#include <stdexcept>

namespace traccc {
namespace details {

struct finding_performance_writer_data {

    /// Constructor
    finding_performance_writer_data(
        const finding_performance_writer::config& cfg)
        : m_eff_plot_tool({cfg.var_binning}),
          m_duplication_plot_tool({cfg.var_binning}),
          m_fake_tracks_plot_tool({cfg.var_binning}),
          m_stat_plot_tool(cfg.stat_config) {}

    /// Plot tool for efficiency
    eff_plot_tool m_eff_plot_tool;
    eff_plot_tool::eff_plot_cache m_eff_plot_cache;

    /// Plot tool for duplication rate
    duplication_plot_tool m_duplication_plot_tool;
    duplication_plot_tool::duplication_plot_cache m_duplication_plot_cache;

    // Plot tool for fake tracks monitoring
    fake_tracks_plot_tool m_fake_tracks_plot_tool;
    fake_tracks_plot_tool::fake_tracks_plot_cache m_fake_tracks_plot_cache;

    std::map<measurement, std::map<particle, std::size_t>>
        m_measurement_particle_map;
    std::map<std::uint64_t, particle> m_particle_map;

    /// Plot tool for statistics
    stat_plot_tool m_stat_plot_tool;
    stat_plot_tool::stat_plot_cache m_stat_plot_cache;

};  // struct finding_performance_writer_data

}  // namespace details

finding_performance_writer::finding_performance_writer(const config& cfg)
    : m_cfg(cfg),
      m_data(std::make_unique<details::finding_performance_writer_data>(cfg)) {

    m_data->m_eff_plot_tool.book(m_cfg.algorithm_name,
                                 m_data->m_eff_plot_cache);
    m_data->m_duplication_plot_tool.book(m_cfg.algorithm_name,
                                         m_data->m_duplication_plot_cache);
    m_data->m_fake_tracks_plot_tool.book(m_cfg.algorithm_name,
                                         m_data->m_fake_tracks_plot_cache);
    m_data->m_stat_plot_tool.book(m_data->m_stat_plot_cache);
}

finding_performance_writer::~finding_performance_writer() {}

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
 * @brief For ambiguity resolution only. Associates each reconstructed track
 * with its measurements.
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

void finding_performance_writer::write_common(
    const std::vector<std::vector<measurement>>& tracks,
    const event_data& evt_data) {

    // Associates truth particle_ids with the number of tracks made entirely of
    // some (or all) of its hits.
    std::map<particle_id, std::size_t> match_counter;

    // Associates truth particle_ids with the number of tracks sharing hits from
    // more than one truth particle.
    std::map<particle_id, std::size_t> fake_counter;

    // Iterate over the tracks.
    const std::size_t n_tracks = tracks.size();

    for (std::size_t i = 0; i < n_tracks; i++) {

        const std::vector<measurement>& found_measurements = tracks[i];

        // Check which particle matches this seed.
        // Input :
        //    - the list of measurements for this track
        //    - the truth particles map
        // Output :
        //    - a list of particles, having for each of them a particle_id and
        //      a count value.
        // If there is only a single truth particle contributing to this track,
        // then increment the match_counter for this truth particle id.
        // If there are at least two particles contributing to the hit list of
        // this track, increment the fake_counter for each truth particle.
        std::vector<particle_hit_count> particle_hit_counts;

        if (!evt_data.m_found_meas_to_ptc_map.empty()) {
            particle_hit_counts = identify_contributing_particles(
                found_measurements, evt_data.m_found_meas_to_ptc_map);
        } else {
            particle_hit_counts = identify_contributing_particles(
                found_measurements, evt_data.m_meas_to_ptc_map);
        }

        const auto major_ptc = particle_hit_counts.at(0).ptc;
        const auto n_major_hits = particle_hit_counts.at(0).hit_counts;

        // Truth measureemnt from the particle
        const std::vector<measurement> truth_measurements =
            evt_data.m_ptc_to_meas_map.at(major_ptc);

        // Consider it being matched if hit counts is larger than the half
        // of the number of measurements
        assert(found_measurements.size() > 0u);
        assert(truth_measurements.size() > 0u);
        const bool reco_matched =
            static_cast<double>(n_major_hits) /
                static_cast<double>(found_measurements.size()) >
            m_cfg.matching_ratio;
        const bool truth_matched =
            static_cast<double>(n_major_hits) /
                static_cast<double>(truth_measurements.size()) >
            m_cfg.matching_ratio;

        if ((!m_cfg.double_matching && reco_matched) ||
            (m_cfg.double_matching && reco_matched && truth_matched)) {
            const auto pid = major_ptc.particle_id;
            match_counter[pid]++;
        } else {
            for (particle_hit_count const& phc : particle_hit_counts) {
                const auto pid = phc.ptc.particle_id;
                fake_counter[pid]++;
            }
        }
    }

    // For each truth particle...
    for (auto const& [pid, ptc] : evt_data.m_particle_map) {

        // Count only charged particles which satisfy pT_cut
        if (ptc.charge == 0 || vector::perp(ptc.momentum) < m_cfg.pT_cut ||
            ptc.vertex[2] < m_cfg.z_min || ptc.vertex[2] > m_cfg.z_max ||
            vector::perp(ptc.vertex) > m_cfg.r_max) {
            continue;
        }

        // Finds how many tracks were made solely by hits from the current truth
        // particle
        bool is_matched = false;
        std::size_t n_matched_seeds_for_particle = 0;
        auto it = match_counter.find(pid);
        if (it != match_counter.end()) {
            is_matched = true;
            n_matched_seeds_for_particle = it->second;
        }

        // Finds how many (fake) tracks were made with at least one hit from the
        // current truth particle
        std::size_t fake_count = 0;
        auto itf = fake_counter.find(pid);
        if (itf != fake_counter.end()) {
            fake_count = itf->second;
        }

        m_data->m_eff_plot_tool.fill(m_data->m_eff_plot_cache, ptc, is_matched);
        m_data->m_duplication_plot_tool.fill(m_data->m_duplication_plot_cache,
                                             ptc,
                                             n_matched_seeds_for_particle - 1);
        m_data->m_fake_tracks_plot_tool.fill(m_data->m_fake_tracks_plot_cache,
                                             ptc, fake_count);
    }
}

/// For track finding
void finding_performance_writer::write(
    const track_candidate_container_types::const_view& track_candidates_view,
    const event_data& evt_data) {

    // Iterate over the tracks.
    track_candidate_container_types::const_device track_candidates(
        track_candidates_view);

    const unsigned int n_tracks = track_candidates.size();

    for (unsigned int i = 0; i < n_tracks; i++) {

        // Fill stat plot
        m_data->m_stat_plot_tool.fill(m_data->m_stat_plot_cache,
                                      track_candidates.at(i).header);
    }

    std::vector<std::vector<measurement>> tracks =
        prepare_data(track_candidates_view);
    write_common(tracks, evt_data);
}

/// For ambiguity resolution
void finding_performance_writer::write(
    const track_state_container_types::const_view& track_states_view,
    const event_data& evt_data) {
    std::vector<std::vector<measurement>> tracks =
        prepare_data(track_states_view);
    write_common(tracks, evt_data);
}

void finding_performance_writer::finalize() {

#ifdef TRACCC_HAVE_ROOT
    // Open the output file.
    std::unique_ptr<TFile> ofile(
        TFile::Open(m_cfg.file_path.c_str(), m_cfg.file_mode.c_str()));
    if ((!ofile) || ofile->IsZombie()) {
        throw std::runtime_error("Could not open output file \"" +
                                 m_cfg.file_path + "\" in mode \"" +
                                 m_cfg.file_mode + "\"");
    }
    ofile->cd();
#else
    std::cout << "ROOT file \"" << m_cfg.file_path << "\" is NOT created"
              << std::endl;
#endif  // TRACCC_HAVE_ROOT

    m_data->m_eff_plot_tool.write(m_data->m_eff_plot_cache);
    m_data->m_duplication_plot_tool.write(m_data->m_duplication_plot_cache);
    m_data->m_fake_tracks_plot_tool.write(m_data->m_fake_tracks_plot_cache);
    m_data->m_stat_plot_tool.write(m_data->m_stat_plot_cache);
}

}  // namespace traccc
