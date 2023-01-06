/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).

#include "traccc/edm/track_state.hpp"
#include "traccc/io/event_map2.hpp"
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
    /// @param det detector object
    /// @param evt_map event map to find the truth values
    template <typename detector_t>
    void write(const track_state_collection_types::host& track_states_per_track,
               const detector_t& det, event_map2& evt_map) {

        auto& m_p_map = evt_map.meas_ptc_map;

        // Get the track state at the first surface
        const auto& trk_state = track_states_per_track[0];
        const measurement_link meas_link{trk_state.surface_link(),
                                         trk_state.get_measurement()};

        // Find the contributing particle
        // @todo: Use identify_contributing_particles function
        std::map<particle, uint64_t> contributing_particles =
            m_p_map[meas_link];

        const particle ptc = contributing_particles.begin()->first;

        // Find the truth global position and momentum
        const auto global_pos = evt_map.meas_xp_map[meas_link].first;
        const auto global_mom = evt_map.meas_xp_map[meas_link].second;

        const auto truth_local = det.global_to_local(
            meas_link.surface_link, global_pos, vector::normalize(global_mom));

        // Return value
        bound_track_parameters truth_param;
        auto& truth_vec = truth_param.vector();
        getter::element(truth_vec, e_bound_loc0, 0) = truth_local[0];
        getter::element(truth_vec, e_bound_loc1, 0) = truth_local[1];
        getter::element(truth_vec, e_bound_phi, 0) = getter::phi(global_mom);
        getter::element(truth_vec, e_bound_theta, 0) =
            getter::theta(global_mom);
        // @todo: Assign a proper value to time
        getter::element(truth_vec, e_bound_time, 0) = 0.;
        getter::element(truth_vec, e_bound_qoverp, 0) =
            ptc.charge / getter::norm(global_mom);

        // For the moment, only fill with the first measurements
        m_res_plot_tool.fill(m_res_plot_cache, truth_param,
                             trk_state.smoothed());
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