/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/efficiency/finding_performance_writer.hpp"

#include "../resolution/stat_plot_tool.hpp"
#include "duplication_plot_tool.hpp"
#include "eff_plot_tool.hpp"
#include "fake_tracks_plot_tool.hpp"
#include "traccc/utils/logging.hpp"
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

finding_performance_writer::finding_performance_writer(
    const config& cfg, std::unique_ptr<const traccc::Logger> logger)
    : messaging(std::move(logger)),
      m_cfg(cfg),
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
    const edm::track_candidate_container<default_algebra>::const_view&
        track_candidates_view) {
    std::vector<std::vector<measurement>> result;

    // Iterate over the tracks.
    const edm::track_candidate_container<default_algebra>::const_device
        track_candidates(track_candidates_view);

    const unsigned int n_tracks = track_candidates.tracks.size();
    result.reserve(n_tracks);

    for (unsigned int i = 0; i < n_tracks; i++) {
        std::vector<measurement> m;
        m.reserve(track_candidates.tracks.at(i).measurement_indices().size());
        const edm::track_candidate_collection<
            default_algebra>::const_device::const_proxy_type track =
            track_candidates.tracks.at(i);
        for (unsigned int midx : track.measurement_indices()) {
            m.push_back(track_candidates.measurements.at(midx));
        }
        result.push_back(std::move(m));
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
    const edm::track_fit_container<default_algebra>::const_view&
        track_fit_view) {
    std::vector<std::vector<measurement>> result;

    // Set up the input containers.
    const edm::track_fit_container<default_algebra>::const_device track_fits(
        track_fit_view);

    // Iterate over the tracks.
    const unsigned int n_tracks = track_fits.tracks.size();
    result.reserve(n_tracks);

    for (unsigned int i = 0; i < n_tracks; i++) {
        std::vector<measurement> result_measurements;
        result_measurements.reserve(
            track_fits.tracks.at(i).state_indices().size());
        for (unsigned int st_idx : track_fits.tracks.state_indices().at(i)) {
            result_measurements.push_back(track_fits.measurements.at(
                track_fits.states.at(st_idx).measurement_index()));
        }
        result.push_back(std::move(result_measurements));
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

    std::size_t total_fake_tracks = 0;

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

        const double purity = static_cast<double>(n_major_hits) /
                              static_cast<double>(found_measurements.size());
        const double completeness =
            static_cast<double>(n_major_hits) /
            static_cast<double>(truth_measurements.size());

        const bool reco_matched =
            purity > m_cfg.track_truth_config.matching_ratio;
        const bool truth_matched =
            completeness > m_cfg.track_truth_config.matching_ratio;

        m_data->m_stat_plot_tool.fill(m_data->m_stat_plot_cache, purity,
                                      completeness);

        if ((!m_cfg.track_truth_config.double_matching && reco_matched) ||
            (m_cfg.track_truth_config.double_matching && reco_matched &&
             truth_matched)) {
            const auto pid = major_ptc.particle_id;
            match_counter[pid]++;
        } else {
            for (particle_hit_count const& phc : particle_hit_counts) {
                const auto pid = phc.ptc.particle_id;
                fake_counter[pid]++;
            }
            total_fake_tracks++;
        }
    }

    std::size_t total_truth_particles = 0;
    std::size_t total_matched_truth_particles = 0;
    std::size_t total_duplicate_tracks = 0;

    // For each truth particle...
    for (auto const& [pid, ptc] : evt_data.m_particle_map) {
        std::size_t num_measurements = 0;

        // Find the number of measurements belonging to this track
        if (auto it = evt_data.m_ptc_to_meas_map.find(ptc);
            it != evt_data.m_ptc_to_meas_map.end()) {
            num_measurements = it->second.size();
        } else {
            continue;
        }

        // Count only charged particles which satisfy pT_cut and vertex cut
        if (ptc.charge == 0 ||
            vector::perp(ptc.momentum) < m_cfg.truth_config.pT_min ||
            ptc.vertex[2] < m_cfg.truth_config.z_min ||
            ptc.vertex[2] > m_cfg.truth_config.z_max ||
            vector::perp(ptc.vertex) > m_cfg.truth_config.r_max ||
            num_measurements < m_cfg.truth_config.min_track_candidates) {
            continue;
        }

        total_truth_particles++;

        // Finds how many tracks were made solely by hits from the current truth
        // particle
        bool is_matched = false;
        std::size_t n_matched_seeds_for_particle = 0;
        auto it = match_counter.find(pid);
        if (it != match_counter.end()) {
            is_matched = true;
            total_matched_truth_particles++;
            n_matched_seeds_for_particle = it->second;
            total_duplicate_tracks += n_matched_seeds_for_particle - 1;
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

    TRACCC_INFO("Total number of truth particles was "
                << total_truth_particles);
    TRACCC_INFO("Total number of found tracks was " << n_tracks);
    TRACCC_INFO("Total number of track-matched particles was "
                << total_matched_truth_particles);
    TRACCC_INFO("Total number of duplicated tracks was "
                << total_duplicate_tracks);
    TRACCC_INFO("Total number of fake tracks was " << total_fake_tracks);
    TRACCC_INFO("Total track efficiency was "
                << (100. * static_cast<double>(total_matched_truth_particles) /
                    static_cast<double>(total_truth_particles))
                << "%");
    TRACCC_INFO("Total track duplicate rate was "
                << (static_cast<double>(total_duplicate_tracks) /
                    static_cast<double>(total_matched_truth_particles)));
    TRACCC_INFO("Total track fake rate was "
                << (static_cast<double>(total_fake_tracks) /
                    static_cast<double>(total_truth_particles)));
}

/// For track finding
void finding_performance_writer::write(
    const edm::track_candidate_container<default_algebra>::const_view&
        track_candidates_view,
    const event_data& evt_data) {

    // Set up the input containers.
    const edm::track_candidate_collection<default_algebra>::const_device
        track_candidates(track_candidates_view.tracks);

    const unsigned int n_tracks = track_candidates.size();

    for (unsigned int i = 0; i < n_tracks; i++) {

        // Fill stat plot
        m_data->m_stat_plot_tool.fill(m_data->m_stat_plot_cache,
                                      track_candidates.at(i));
    }

    std::vector<std::vector<measurement>> tracks =
        prepare_data(track_candidates_view);
    write_common(tracks, evt_data);
}

/// For ambiguity resolution
void finding_performance_writer::write(
    const edm::track_fit_container<default_algebra>::const_view& track_fit_view,
    const event_data& evt_data) {

    std::vector<std::vector<measurement>> tracks = prepare_data(track_fit_view);
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
