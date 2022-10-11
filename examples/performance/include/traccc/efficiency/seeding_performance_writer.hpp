/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/seed.hpp"
#include "traccc/efficiency/duplication_plot_tool.hpp"
#include "traccc/efficiency/eff_plot_tool.hpp"
#include "traccc/efficiency/track_classification.hpp"
#include "traccc/event/event_map.hpp"
#include "traccc/io/mapper.hpp"

// ROOT
#include <TFile.h>
#include <TTree.h>

namespace traccc {

class seeding_performance_writer {

    public:
    struct config {

        /// Output filename.
        std::string file_path = "performance_track_seeding.root";
        /// Output file mode
        std::string file_mode = "RECREATE";

        /// Plot tool configurations.
        eff_plot_tool::config eff_plot_tool_config;
        duplication_plot_tool::config duplication_plot_tool_config;

        /// Cut values
        scalar pT_cut = 1.;
    };

    /// Construct from configuration and log level.
    /// @param config The configuration
    /// @param level
    seeding_performance_writer(config cfg)
        : m_cfg(std::move(cfg)),
          m_eff_plot_tool(m_cfg.eff_plot_tool_config),
          m_duplication_plot_tool(m_cfg.duplication_plot_tool_config){};

    ~seeding_performance_writer() {

        for (auto& [name, cache] : m_eff_plot_caches) {
            m_eff_plot_tool.clear(cache);
        }

        for (auto& [name, cache] : m_duplication_plot_caches) {
            m_duplication_plot_tool.clear(cache);
        }

        if (m_output_file) {
            m_output_file->Close();
        }
    }

    void add_cache(std::string name) {
        if (not m_output_file) {
            init();
        }

        m_eff_plot_tool.book(name, m_eff_plot_caches[name]);
        m_duplication_plot_tool.book(name, m_duplication_plot_caches[name]);
    }

    void write(std::string name, const seed_collection_types::host& seeds,
               const spacepoint_container_types::host& spacepoints,
               event_map& evt_map) {

        auto& p_map = evt_map.ptc_map;
        auto& m_p_map = evt_map.meas_ptc_map;

        std::map<particle_id, std::size_t> match_counter;

        std::size_t n_matched_seeds = 0;

        for (const auto& sd : seeds) {
            auto measurements = sd.get_measurements(get_data(spacepoints));

            std::vector<particle_hit_count> particle_hit_counts =
                identify_contributing_particles(measurements, m_p_map);

            if (particle_hit_counts.size() == 1) {
                auto pid = particle_hit_counts.at(0).ptc.particle_id;

                match_counter[pid]++;
                n_matched_seeds++;
            }
        }

        std::size_t n_particles = 0;
        std::size_t n_matched_particles = 0;
        std::size_t n_duplicated_particles = 0;

        for (auto const& [pid, ptc] : p_map) {
            bool is_matched = false;
            auto it = match_counter.find(pid);

            std::size_t n_matched_seeds_for_particle = 0;

            // Count only charged particles which satisfiy pT_cut
            if (ptc.charge == 0 || getter::perp(ptc.mom) < m_cfg.pT_cut) {
                continue;
            }

            n_particles++;

            if (it != match_counter.end()) {
                is_matched = true;
                n_matched_particles++;
                n_matched_seeds_for_particle = it->second;

                if (n_matched_seeds_for_particle > 1) {
                    n_duplicated_particles++;
                }
            }

            m_eff_plot_tool.fill(m_eff_plot_caches[name], ptc, is_matched);
            m_duplication_plot_tool.fill(m_duplication_plot_caches[name], ptc,
                                         n_matched_seeds_for_particle - 1);
        }

        m_n_total_seeds += seeds.size();
        m_n_total_matched_seeds += n_matched_seeds;
        m_n_total_particles += n_particles;
        m_n_total_matched_particles += n_matched_particles;
        m_n_total_duplicated_particles = n_duplicated_particles;
    }

    void finalize() {
        m_output_file->cd();

        for (auto const& [name, cache] : m_eff_plot_caches) {
            m_eff_plot_tool.write(cache);
        }

        for (auto const& [name, cache] : m_duplication_plot_caches) {
            m_duplication_plot_tool.write(cache);
        }
    }

    private:
    void init() {
        const std::string& path = m_cfg.file_path;

        m_output_file = std::unique_ptr<TFile>{
            TFile::Open(path.c_str(), m_cfg.file_mode.c_str())};

        if (not m_output_file) {
            throw std::invalid_argument("Could not open '" + path + "'");
        }
    }

    config m_cfg;

    std::unique_ptr<TFile> m_output_file{nullptr};
    /// Plot tool for efficiency
    eff_plot_tool m_eff_plot_tool;
    std::map<std::string, eff_plot_tool::eff_plot_cache> m_eff_plot_caches;

    /// Plot tool for duplication rate
    duplication_plot_tool m_duplication_plot_tool;
    std::map<std::string, duplication_plot_tool::duplication_plot_cache>
        m_duplication_plot_caches;

    size_t m_n_total_seeds = 0;
    size_t m_n_total_matched_seeds = 0;
    size_t m_n_total_particles = 0;
    size_t m_n_total_matched_particles = 0;
    size_t m_n_total_duplicated_particles = 0;

    measurement_particle_map m_measurement_particle_map;
    particle_map m_particle_map;
};

}  // namespace traccc
