/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/resolution/res_plot_tool_config.hpp"
#include "traccc/resolution/stat_plot_tool_config.hpp"

// Project include(s).
#include "traccc/edm/particle.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/io/event_map2.hpp"
#include "traccc/io/mapper.hpp"

// System include(s).
#include <memory>

namespace traccc {
namespace details {

/// Data members that should not pollute the API of
/// @c traccc::fitting_performance_writer
struct fitting_performance_writer_data;

}  // namespace details

class fitting_performance_writer {

    public:
    struct config {
        /// Output filename.
        std::string file_path = "performance_track_fitting.root";
        /// Output file mode
        std::string file_mode = "RECREATE";
        /// Plot tool configurations.
        res_plot_tool_config res_config;
        stat_plot_tool_config stat_config;
    };

    /// Constructor with writer config
    fitting_performance_writer(const config& cfg);

    /// Destructor that closes the file
    ~fitting_performance_writer();

    /// Fill the tracking results into the histograms
    ///
    /// @param track_states_per_track vector of track states of a track
    /// @param det detector object
    /// @param evt_map event map to find the truth values
    template <typename detector_t>
    void write(const track_state_collection_types::host& track_states_per_track,
               const fitting_result<traccc::default_algebra>& fit_res,
               const detector_t& det, event_map2& evt_map) {

        auto& m_p_map = evt_map.meas_ptc_map;

        // Get the track state at the first surface
        const auto& trk_state = track_states_per_track[0];
        const measurement meas = trk_state.get_measurement();

        // Find the contributing particle
        // @todo: Use identify_contributing_particles function
        std::map<particle, uint64_t> contributing_particles = m_p_map[meas];

        const particle ptc = contributing_particles.begin()->first;

        // Find the truth global position and momentum
        const auto global_pos = evt_map.meas_xp_map[meas].first;
        const auto global_mom = evt_map.meas_xp_map[meas].second;

        const detray::tracking_surface sf{det, meas.surface_link};
        using cxt_t = typename detector_t::geometry_context;
        const cxt_t ctx{};
        const auto truth_local =
            sf.global_to_local(ctx, global_pos, vector::normalize(global_mom));

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
        if (fit_res.ndf > 0 && !trk_state.is_hole) {
            write_res(truth_param, trk_state.smoothed(), ptc);
        }
        write_stat(fit_res, track_states_per_track);
    }

    /// Writing caches into the file
    void finalize();

    private:
    /// Non-templated part of the @c write(...) function
    void write_res(const bound_track_parameters& truth_param,
                   const bound_track_parameters& fit_param,
                   const particle& ptc);

    /// Non-templated part of the @c write(...) function
    void write_stat(const fitting_result<traccc::default_algebra>& fit_res,
                    const track_state_collection_types::host& track_states);

    /// Configuration for the tool
    config m_cfg;

    /// Opaque data members for the class
    std::unique_ptr<details::fitting_performance_writer_data> m_data;

};  // class fitting_performance_writer

}  // namespace traccc
