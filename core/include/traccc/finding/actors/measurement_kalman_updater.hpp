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
#include "traccc/finding/finding_config.hpp"
#include "traccc/finding/measurement_selector.hpp"
#include "traccc/finding/track_state_candidate.hpp"
#include "traccc/fitting/kalman_filter/gain_matrix_updater.hpp"
#include "traccc/fitting/kalman_filter/is_line_visitor.hpp"
#include "traccc/fitting/status_codes.hpp"
#include "traccc/utils/logging.hpp"

// detray include(s)
#include <detray/propagator/base_actor.hpp>
#include <detray/utils/ranges/detail/iterator_functions.hpp>

namespace traccc {

/// Some statistics of a given candidate track
template <detray::concepts::scalar scalar_t>
struct track_stats {
    scalar_t chi2_sum{0.f};
    unsigned int seed_idx{std::numeric_limits<unsigned int>::max()};
    std::uint_least16_t n_track_states{0u};
    std::uint_least16_t ndf_sum{0u};
    std::uint_least16_t n_holes{0u};
    std::uint_least16_t n_consecutive_holes{0u};
};

/// Find the optimal next measurement and perform a Kalman update on it
template <detray::concepts::algebra algebra_t>
struct measurement_updater : detray::base_actor {

    /// Contains the current track states and some statistics
    struct state {
        using scalar_t = detray::dscalar<algebra_t>;
        using measurement_collection_t = edm::measurement_collection<algebra_t>;

        constexpr state() = default;

        TRACCC_HOST_DEVICE
        constexpr state(
            typename edm::measurement_collection<algebra_t>::const_device
                measurements,
            vecmem::data::vector_view<unsigned int> meas_ranges_view,
            void* track_state_candidates, const smoother_type smoother)
            : m_measurements{measurements},
              m_measurement_ranges{meas_ranges_view},
              m_cand_ptr{track_state_candidates},
              m_run_smoother{smoother} {}

        /// Congfig params
        scalar_t max_chi2{0.f};
        /// Max no. of track states this actor can accumulate
        unsigned int max_n_track_states{100u};
        /// Max no. holes on track
        unsigned short max_n_holes{3u};
        /// Max no. consecutive holes on track
        unsigned short max_n_consecutive_holes{2u};

        /// Statistics for the current track
        track_stats<scalar_t> m_stats{};

        /// Calibration configuration
        traccc::measurement_selector::config m_calib_cfg{};

        /// Measurement container
        typename measurement_collection_t::const_device m_measurements{
            typename measurement_collection_t::const_view{}};
        /// Per surface measurement index ranges into measurement cont.
        vecmem::device_vector<unsigned int> m_measurement_ranges{
            vecmem::data::vector_view<unsigned int>{}};
        /// Track states for the current track
        void* m_cand_ptr{nullptr};
        /// The track candidate collection type pointed to
        smoother_type m_run_smoother{smoother_type::e_mbf};
    };

    /// Select the optimal next measurement and run the KF update
    template <typename transporter_result_t, typename propagator_state_t>
    TRACCC_HOST_DEVICE void operator()(state& updater_state,
                                       transporter_result_t& transporter_result,
                                       propagator_state_t& propagation) const {
        using scalar_t = detray::dscalar<algebra_t>;

        auto& navigation = propagation._navigation;
        auto& stepping = propagation._stepping;

        // Measurements are only available on sensitive surfaces
        if (!navigation.is_on_sensitive()) {
            return;
        }

        TRACCC_VERBOSE_HOST_DEVICE(
            "Actor: Update track parameters with new measurement...");

        // Get current detector surface (sensitive)
        const auto sf = navigation.current_surface();
        const bool is_line = detail::is_line(sf);
        assert(sf.is_sensitive());

        // Track parameters on the sensitive surface
        auto& bound_param = transporter_result.destination_params;

        TRACCC_DEBUG_HOST("-> Predicted param.:\n" << bound_param);
        TRACCC_VERBOSE_HOST_DEVICE("-> Calculate predicted chi2:");

        // Find the measurement with the smallest predicted chi2
        const auto& measurements = updater_state.m_measurements;
        const candidate_measurement cand =
            measurement_selector::find_optimal_measurement(
                bound_param, measurements, updater_state.m_measurement_ranges,
                updater_state.m_calib_cfg, is_line);
        if (cand.chi2 < std::numeric_limits<scalar_t>::max()) {
            TRACCC_VERBOSE_HOST_DEVICE(
                "Optimal measurement: %d (pred. chi2 = %f)", cand.meas_idx,
                cand.chi2);
        } else {
            TRACCC_VERBOSE_HOST_DEVICE("No measurement found");
        }

        scalar_t filtered_chi2{0.f};

        // Run the KF update and add the track state
        const scalar_t max_chi2{
            navigation.is_edge_candidate() ? 1.f : updater_state.max_chi2};
        if (cand.chi2 <= max_chi2) {
            const auto meas = measurements.at(cand.meas_idx);

            kalman_fitter_status res{kalman_fitter_status::ERROR_OTHER};

            // No Kalman update needed (first measurement of the seed)
            if (cand.chi2 <= std::numeric_limits<scalar_t>::epsilon() &&
                !sf.has_material()) {

                // Update measurement covariance
                const auto V =
                    measurement_selector::calibrated_measurement_covariance<
                        algebra_t, 2>(meas, updater_state.m_calib_cfg);

                auto& filtered_cov = bound_param.covariance();
                getter::element(filtered_cov, e_bound_loc0, e_bound_loc0) =
                    getter::element(V, 0, 0);
                getter::element(filtered_cov, e_bound_loc1, e_bound_loc1) =
                    getter::element(V, 1, 1);

                TRACCC_DEBUG_HOST("-> Updated track parameters:\n"
                                  << bound_param);

                res = kalman_fitter_status::SUCCESS;
            } else {
                // Run the Kalman update on the bound track params (in place!)
                constexpr gain_matrix_updater<algebra_t> kalman_updater{};
                res = kalman_updater(bound_param, filtered_chi2, meas,
                                     bound_param, is_line);

                TRACCC_DEBUG_HOST("-> KF status: " << fitter_debug_msg{res}());
                // Abandon measurement in case of filter failure
                if (res != kalman_fitter_status::SUCCESS) {
                    TRACCC_ERROR_HOST(
                        "KF failure: " << fitter_debug_msg{res}());
                    TRACCC_ERROR_DEVICE("KF failure: %d",
                                        static_cast<int>(res));

                    TRACCC_WARNING_HOST_DEVICE("Counting this as hole!");
                    updater_state.m_stats.n_holes++;
                    updater_state.m_stats.n_consecutive_holes++;

                    if (bound_param.is_invalid()) {
                        navigation.exit();
                        propagation._heartbeat = false;
                        return;
                    }
                }

                propagation.set_particle(detail::correct_particle_hypothesis(
                    stepping.particle_hypothesis(), bound_param));

                // Consistency check down to 1% relative deviation
                if (!algebra::approx_equal(filtered_chi2, cand.chi2, 0.01f)) {
                    TRACCC_WARNING_HOST_DEVICE(
                        "Chi2 deviation! predicted: %f, filtered: %f",
                        cand.chi2, filtered_chi2);
                }

                // Flag renavigation of the current candidate (unless overlap)
                if (math::fabs(navigation()) > 1.f * unit<float>::um) {
                    navigation.set_high_trust();
                } else {
                    TRACCC_DEBUG_HOST_DEVICE(
                        "-> Encountered overlap, jump to next surface");
                }
            }

            TRACCC_VERBOSE_HOST_DEVICE("Assigned measurement: %d",
                                       cand.meas_idx);

            // TODO: Get host-device compatible visitor implementation
            const auto i{
                static_cast<int>(updater_state.m_stats.n_track_states)};
            if (updater_state.m_run_smoother == smoother_type::e_none) {
                auto* track_cand_ptr = static_cast<track_state_candidate*>(
                    updater_state.m_cand_ptr);

                detray::ranges::detail::advance(track_cand_ptr, i);

                assert(track_cand_ptr);
                *track_cand_ptr = {cand.meas_idx};
            } else if (updater_state.m_run_smoother ==
                       smoother_type::e_kalman) {
                auto* track_cand_ptr =
                    static_cast<filtered_track_state_candidate<algebra_t>*>(
                        updater_state.m_cand_ptr);

                detray::ranges::detail::advance(track_cand_ptr, i);

                assert(track_cand_ptr);
                *track_cand_ptr = {cand.meas_idx, filtered_chi2, bound_param};
            } else if (updater_state.m_run_smoother == smoother_type::e_mbf) {
                auto* track_cand_ptr =
                    static_cast<full_track_state_candidate<algebra_t>*>(
                        updater_state.m_cand_ptr);

                detray::ranges::detail::advance(track_cand_ptr, i);

                // TODO: Get proper Jacobian
                traccc::bound_track_parameters<algebra_t> predicted_params{};
                traccc::bound_matrix<algebra_t> full_jac{};

                assert(track_cand_ptr);
                *track_cand_ptr = {cand.meas_idx, filtered_chi2, bound_param,
                                   predicted_params, full_jac};
            } else {
                navigation.abort(
                    "Unknown data coll. type in measurement updater");
                propagation._heartbeat = false;
                return;
            }

            // The updated track params. should be published to the propagation
            transporter_result.status = detray::actor::status::e_success;

            // Update statistics
            updater_state.m_stats.n_consecutive_holes = 0u;
            updater_state.m_stats.ndf_sum +=
                static_cast<std::uint_least16_t>(meas.dimensions());
            updater_state.m_stats.chi2_sum += filtered_chi2;
            updater_state.m_stats.n_track_states++;

            if (updater_state.m_stats.n_track_states >=
                updater_state.max_n_track_states) {
                TRACCC_WARNING_HOST_DEVICE(
                    "Max. number of track states reached");
                navigation.exit();
                propagation._heartbeat = false;
                return;
            }
        } else {
            // If the surface was only hit due to tolerances, don't count holes
            if (!navigation.is_edge_candidate()) {
                TRACCC_WARNING_HOST_DEVICE(
                    "Found hole: Continue propagation without update");
                updater_state.m_stats.n_holes++;
                updater_state.m_stats.n_consecutive_holes++;
            }
            // If the total number of holes is too large, exit
            if (updater_state.m_stats.n_holes > updater_state.max_n_holes) {
                TRACCC_WARNING_HOST_DEVICE("Maximum total number of holes");
                navigation.exit();
                propagation._heartbeat = false;
            }
            // If the number of consecutive holes is too large, exit
            if (updater_state.m_stats.n_consecutive_holes >
                updater_state.max_n_consecutive_holes) {
                TRACCC_WARNING_HOST_DEVICE(
                    "Maximum number of consecutive holes");
                navigation.exit();
                propagation._heartbeat = false;
            }

            // Discasrd the current local track parameters
            transporter_result.status = detray::actor::status::e_failure;
        }
    }
};

}  // namespace traccc
