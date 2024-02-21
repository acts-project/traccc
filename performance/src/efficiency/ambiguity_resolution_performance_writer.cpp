/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/efficiency/ambiguity_resolution_performance_writer.hpp"

#include "duplication_plot_tool.hpp"
#include "eff_plot_tool.hpp"
#include "track_classification.hpp"

// ROOT include(s).
#ifdef TRACCC_HAVE_ROOT
#include <TFile.h>
#endif  // TRACCC_HAVE_ROOT

// System include(s).
#include <iostream>
#include <memory>
#include <stdexcept>

// Adapted from finding_performance_writer.cpp

namespace traccc {
namespace details {

struct ambiguity_resolution_performance_writer_data {

    /// Constructor
    ambiguity_resolution_performance_writer_data(
        const ambiguity_resolution_performance_writer::config& cfg)
        : m_eff_plot_tool({cfg.var_binning}),
          m_duplication_plot_tool({cfg.var_binning}) {}

    /// Plot tool for efficiency
    eff_plot_tool m_eff_plot_tool;
    eff_plot_tool::eff_plot_cache m_eff_plot_cache;

    /// Plot tool for duplication rate
    duplication_plot_tool m_duplication_plot_tool;
    duplication_plot_tool::duplication_plot_cache m_duplication_plot_cache;

    measurement_particle_map m_measurement_particle_map;
    particle_map m_particle_map;

};  // struct ambiguity_resolution_performance_writer_data

}  // namespace details

ambiguity_resolution_performance_writer::
    ambiguity_resolution_performance_writer(const config& cfg)
    : m_cfg(cfg),
      m_data(std::make_unique<
             details::ambiguity_resolution_performance_writer_data>(cfg)) {

    m_data->m_eff_plot_tool.book("ambiguity_resolution",
                                 m_data->m_eff_plot_cache);
    m_data->m_duplication_plot_tool.book("ambiguity_resolution",
                                         m_data->m_duplication_plot_cache);
}

ambiguity_resolution_performance_writer::
    ~ambiguity_resolution_performance_writer() {}

void ambiguity_resolution_performance_writer::write(
    const track_state_container_types::const_view& track_states_view,
    const event_map2& evt_map) {

    std::map<particle_id, std::size_t> match_counter;

    // Iterate over the tracks.
    track_state_container_types::const_device track_states(track_states_view);

    const unsigned int n_tracks = track_states.size();
    for (unsigned int track_index = 0; track_index < n_tracks; ++track_index) {

        auto const& [fit_res, states] = track_states.at(track_index);

        std::vector<measurement> measurements;
        measurements.reserve(states.size());
        for (const auto& st : states) {
            measurements.push_back(st.get_measurement());
        }

        // Check which particle matches this seed.
        std::vector<particle_hit_count> particle_hit_counts =
            identify_contributing_particles(measurements, evt_map.meas_ptc_map);

        if (particle_hit_counts.size() == 1) {
            auto pid = particle_hit_counts.at(0).ptc.particle_id;
            match_counter[pid]++;
        }
    }

    for (auto const& [pid, ptc] : evt_map.ptc_map) {

        // Count only charged particles which satisfiy pT_cut
        if (ptc.charge == 0 || getter::perp(ptc.mom) < m_cfg.pT_cut) {
            continue;
        }

        bool is_matched = false;
        std::size_t n_matched_seeds_for_particle = 0;
        auto it = match_counter.find(pid);
        if (it != match_counter.end()) {
            is_matched = true;
            n_matched_seeds_for_particle = it->second;
        }

        m_data->m_eff_plot_tool.fill(m_data->m_eff_plot_cache, ptc, is_matched);
        m_data->m_duplication_plot_tool.fill(m_data->m_duplication_plot_cache,
                                             ptc,
                                             n_matched_seeds_for_particle - 1);
    }
}

void ambiguity_resolution_performance_writer::finalize() {

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
}

}  // namespace traccc
