/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
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
#include "traccc/utils/event_data.hpp"

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
               const detector_t& det, event_data& evt_data) {

        static_assert(std::same_as<typename detector_t::algebra_type,
                                   traccc::default_algebra>);

        if (fit_res.fit_outcome != fitter_outcome::SUCCESS) {
            return;
        }

        // Get the first smoothed track state
        const auto& trk_state = *std::find_if(
            track_states_per_track.begin(), track_states_per_track.end(),
            [](const auto& st) { return st.is_smoothed; });
        assert(!trk_state.is_hole);
        assert(trk_state.is_smoothed);

        std::map<measurement, std::map<particle, std::size_t>> meas_to_ptc_map;
        std::map<measurement, std::pair<point3, point3>> meas_to_param_map;

        if (!evt_data.m_found_meas_to_ptc_map.empty()) {
            meas_to_ptc_map = evt_data.m_found_meas_to_ptc_map;
            meas_to_param_map = evt_data.m_found_meas_to_param_map;
        } else {
            meas_to_ptc_map = evt_data.m_meas_to_ptc_map;
            meas_to_param_map = evt_data.m_meas_to_param_map;
        }

        const measurement meas = trk_state.get_measurement();

        // Find the contributing particle
        // @todo: Use identify_contributing_particles function
        std::map<particle, std::size_t> contributing_particles =
            meas_to_ptc_map.at(meas);

        const particle ptc = contributing_particles.begin()->first;

        // Find the truth global position and momentum
        const auto global_pos = meas_to_param_map.at(meas).first;
        const auto global_mom = meas_to_param_map.at(meas).second;

        const detray::tracking_surface sf{det, meas.surface_link};
        using cxt_t = typename detector_t::geometry_context;
        const cxt_t ctx{};
        const auto truth_bound =
            sf.global_to_bound(ctx, global_pos, vector::normalize(global_mom));

        // Return value
        bound_track_parameters<> truth_param{};
        truth_param.set_bound_local(truth_bound);
        truth_param.set_phi(vector::phi(global_mom));
        truth_param.set_theta(vector::theta(global_mom));
        // @todo: Assign a proper value to time
        truth_param.set_time(0.f);
        truth_param.set_qop(ptc.charge / vector::norm(global_mom));

        // For the moment, only fill with the first measurements
        write_res(truth_param, trk_state.smoothed(), ptc);
        write_stat(fit_res, track_states_per_track);
    }

    /// Writing caches into the file
    void finalize();

    private:
    /// Non-templated part of the @c write(...) function
    void write_res(const bound_track_parameters<>& truth_param,
                   const bound_track_parameters<>& fit_param,
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
