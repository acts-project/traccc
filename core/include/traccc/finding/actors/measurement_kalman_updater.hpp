/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/definitions/track_parametrization.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/measurement_helpers.hpp"
#include "traccc/edm/track_state_collection.hpp"
#include "traccc/finding/measurement_selector.hpp"
#include "traccc/fitting/kalman_filter/gain_matrix_updater.hpp"
#include "traccc/fitting/kalman_filter/is_line_visitor.hpp"
#include "traccc/fitting/status_codes.hpp"
#include "traccc/utils/logging.hpp"

// detray include(s)
#include <detray/propagator/base_actor.hpp>

namespace traccc {

/// Some statistics of a given candidate track
template <detray::concepts::scalar scalar_t>
struct track_stats {
    scalar_t ndf_sum{0.f};
    scalar_t chi2_sum{0.f};
    unsigned short n_holes{0u};
    unsigned short n_consecutive_holes{0u};
};

/// Find the optimal next measurement and perform a Kalman update on it
template <detray::concepts::algebra algebra_t>
struct measurement_updater : detray::actor {

    // Contains the current track states and some statistics
    struct state {
        using scalar_t = detray::dscalar<algebra_t>;
        using measurement_collection_t = edm::measurement_collection<algebra_t>;

        TRACCC_HOST_DEVICE
        constexpr state(
            typename edm::measurement_collection<algebra_t>::const_device
                measurements,
            vecmem::data::vector_view<unsigned int> meas_ranges_view,
            typename edm::track_state_collection<algebra_t>::device&
                track_states)
            : m_measurements{measurements},
              m_measurement_ranges{meas_ranges_view},
              m_track_states{track_states} {}

        // Congfig params
        scalar_t max_chi2{0.f};
        // Max no. holes on track
        unsigned short max_n_holes{3u};
        // Max no. consecutive holes on track
        unsigned short max_n_consecutive_holes{2u};

        // Statistics for the current track
        track_stats<scalar_t> m_stats{};

        // Calibration configuration
        traccc::measurement_selector::config m_calib_cfg{};

        // Measurement container
        typename measurement_collection_t::const_device m_measurements;
        // Per surface measurement index ranges into measurement cont.
        vecmem::device_vector<unsigned int> m_measurement_ranges;
        // Track states for the current track
        typename edm::track_state_collection<algebra_t>::device m_track_states;
    };

    /// Select the optimal next measurement and run the KF update
    template <typename propagator_state_t>
    TRACCC_HOST_DEVICE void operator()(state& updater_state,
                                       propagator_state_t& propagation) const {
        auto& navigation = propagation._navigation;
        auto& stepping = propagation._stepping;

        // Measurements are only available on sensitive surfaces
        if (!navigation.is_on_sensitive()) {
            return;
        }

        TRACCC_VERBOSE_HOST_DEVICE("In Kalman measurement updater...");

        // Get current detector surface (sensitive)
        const auto sf = navigation.current_surface();
        const bool is_line = detail::is_line(sf);
        assert(sf.is_sensitive());

        // Track parameters on the sensitive surface
        auto& bound_param = stepping.bound_params();

        // Find the measurement with the smallest predicted chi2
        const auto& measurements = updater_state.m_measurements;
        const candidate_measurement cand =
            measurement_selector::find_optimal_measurement(
                bound_param, measurements, updater_state.m_measurement_ranges,
                updater_state.m_calib_cfg, is_line);

        // Run the KF update and add the track state
        if (cand.chi2 <= updater_state.max_chi2) {
            const unsigned int meas_idx{cand.meas_idx};
            const auto meas = measurements.at(meas_idx);

            auto trk_state =
                edm::make_track_state<algebra_t>(measurements, meas_idx);

            kalman_fitter_status res{kalman_fitter_status::ERROR_OTHER};

            // No Kalman update needed (first measurement of the seed)
            if (cand.chi2 == 0.f && !sf.has_material()) {
                // Update measurement covariance
                const auto V =
                    measurement_selector::calibrated_measurement_covariance<
                        algebra_t, 2>(meas, updater_state.m_calib_cfg);

                auto& filtered_cov = bound_param.covariance();
                getter::element(filtered_cov, e_bound_loc0, e_bound_loc0) =
                    getter::element(V, 0, 0);
                getter::element(filtered_cov, e_bound_loc1, e_bound_loc1) =
                    getter::element(V, 1, 1);

                trk_state.filtered_params() = bound_param;
                trk_state.filtered_chi2() = 0.f;

                TRACCC_VERBOSE_HOST("Updated track parameters:\n"
                                    << bound_param);

                res = kalman_fitter_status::SUCCESS;
            } else {
                // Run the Kalman update on the track state
                constexpr gain_matrix_updater<algebra_t> kalman_updater{};
                res = kalman_updater(trk_state, measurements, bound_param,
                                     is_line);

                TRACCC_DEBUG_HOST("KF status: " << fitter_debug_msg{res}());
                // Abandon measurement in case of filter failure
                if (res != kalman_fitter_status::SUCCESS) {
                    TRACCC_ERROR_HOST(
                        "KF failure: " << fitter_debug_msg{res}());
                    TRACCC_ERROR_DEVICE("KF failure: %d",
                                        static_cast<int>(res));
                    navigation.abort(fitter_debug_msg{res});
                    return;
                }

                // Update propagation on filtered track params
                bound_param = trk_state.filtered_params();
                TRACCC_VERBOSE_HOST("Updated track parameters:\n"
                                    << bound_param);

                // Flag renavigation of the current candidate (unless overlap)
                if (math::fabs(navigation()) > 1.f * unit<float>::um) {
                    navigation.set_high_trust();
                } else {
                    TRACCC_DEBUG_HOST_DEVICE(
                        "Encountered overlap, jump to next surface");
                }
            }

            TRACCC_VERBOSE_HOST_DEVICE("Found measurement: %d", meas_idx);

            // Add the track state to the track
            trk_state.set_hole(false);
            updater_state.m_track_states.push_back(trk_state);

            // Update statistics
            updater_state.m_stats.n_consecutive_holes = 0u;
            updater_state.m_stats.ndf_sum +=
                static_cast<float>(meas.dimensions());
            updater_state.m_stats.chi2_sum += trk_state.filtered_chi2();
        } else {
            // If the surface was only hit due to tolerances, don't count holes
            if (!navigation.is_edge_candidate()) {
                TRACCC_VERBOSE_HOST_DEVICE("Found hole!");
                updater_state.m_stats.n_holes++;
                updater_state.m_stats.n_consecutive_holes++;
            }
            // If the total number of holes is too large, abort
            if (updater_state.m_stats.n_holes > updater_state.max_n_holes) {
                navigation.abort("Maximum total number of holes");
            }
            // If the number of consecutive holes is too large, abort
            if (updater_state.m_stats.n_consecutive_holes >
                updater_state.max_n_consecutive_holes) {
                navigation.abort("Maximum number of consecutive holes");
            }
        }
    }
};

}  // namespace traccc
