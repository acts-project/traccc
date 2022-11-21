/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).

#include "traccc/edm/track_state.hpp"
#include "traccc/event/event_map2.hpp"
#include "traccc/io/mapper.hpp"
#include "traccc/resolution/res_plot_tool.hpp"

// ROOT include(s).
#include <TFile.h>
#include <TTree.h>

namespace traccc {

class fitting_performance_writer {

    public:
    struct config {
        /// Output filename.
        std::string file_path = "performance_track_fitting.root";
        /// Output file mode
        std::string file_mode = "RECREATE";
        /// Plot tool configurations.
        res_plot_tool::config res_plot_tool_config;
    };

    /// Constructor with writer config
    fitting_performance_writer(config cfg);

    /// Destructor that closes the file
    ~fitting_performance_writer();

    /// Fill the tracking results into the histograms
    ///
    /// @param track_states_per_track vector of track states of a track
    /// @param evt_map event map to find the truth values
    template <typename event_store_t>
    void write(const track_state_collection_types::host& track_states_per_track,
               event_store_t& evt_map) {
        // Find truth parameter associated with the track
        const auto truth_param = evt_map.find_truth_param(
            track_states_per_track[0].surface_link(),
            track_states_per_track[0].get_measurement());

        // For the moment, only fill with the first measurements
        m_res_plot_tool.fill(m_res_plot_cache, truth_param,
                             track_states_per_track[0].smoothed());
    }

    /// Writing caches into the file
    void finalize();

    /// Return the file pointer
    ///
    /// @return the file pointer
    TFile* get_file() { return m_output_file.get(); }

    private:
    config m_cfg;
    std::unique_ptr<TFile> m_output_file{nullptr};

    /// Plot tool for resolution
    res_plot_tool m_res_plot_tool;
    res_plot_tool::res_plot_cache m_res_plot_cache;
};

}  // namespace traccc