/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "kalman_actor.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_fit_collection.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/edm/track_state_collection.hpp"
#include "traccc/fitting/fitting_config.hpp"
#include "traccc/fitting/kalman_filter/kalman_actor.hpp"
#include "traccc/fitting/kalman_filter/kalman_step_aborter.hpp"
#include "traccc/fitting/kalman_filter/statistics_updater.hpp"
#include "traccc/fitting/kalman_filter/two_filters_smoother.hpp"
#include "traccc/fitting/status_codes.hpp"
#include "traccc/utils/particle.hpp"
#include "traccc/utils/prob.hpp"
#include "traccc/utils/propagation.hpp"

// vecmem include(s)
#include <type_traits>
#include <vecmem/containers/device_vector.hpp>

// System include(s).
#include <limits>

namespace traccc {

/// Kalman fitter algorithm to fit a single track
template <typename stepper_t, typename navigator_t>
class kalman_fitter {

    public:
    // Detector type
    using detector_type = typename navigator_t::detector_type;

    // Algebra type
    using algebra_type = typename detector_type::algebra_type;

    // scalar type
    using scalar_type = detray::dscalar<algebra_type>;

    /// Configuration type
    using config_type = fitting_config;

    // Field type
    using bfield_type = typename stepper_t::magnetic_field_type;

    // Actor types
    using aborter = detray::pathlimit_aborter<scalar_type>;
    using transporter = detray::parameter_transporter<algebra_type>;
    using interactor = detray::pointwise_material_interactor<algebra_type>;
    using forward_fit_actor =
        traccc::kalman_actor<algebra_type,
                             kalman_actor_direction::FORWARD_ONLY>;
    using backward_fit_actor =
        traccc::kalman_actor<algebra_type,
                             kalman_actor_direction::BACKWARD_ONLY>;
    using resetter = detray::parameter_resetter<algebra_type>;
    using barcode_sequencer = detray::barcode_sequencer;

    static_assert(std::is_same_v<typename forward_fit_actor::state,
                                 typename backward_fit_actor::state>);

    using forward_actor_chain_type =
        detray::actor_chain<aborter, transporter, interactor, forward_fit_actor,
                            resetter, barcode_sequencer, kalman_step_aborter>;

    using backward_actor_chain_type =
        detray::actor_chain<aborter, transporter, backward_fit_actor,
                            interactor, resetter, kalman_step_aborter>;

    // Navigator type for backward propagator
    using direct_navigator_type = detray::direct_navigator<detector_type>;

    // Propagator type
    using forward_propagator_type =
        detray::propagator<stepper_t, navigator_t, forward_actor_chain_type>;

    using backward_propagator_type =
        detray::propagator<stepper_t, direct_navigator_type,
                           backward_actor_chain_type>;

    /// Constructor with a detector
    ///
    /// @param det the detector object
    TRACCC_HOST_DEVICE
    kalman_fitter(const detector_type& det, const bfield_type& field,
                  const config_type& cfg)
        : m_detector(det), m_field(field), m_cfg(cfg) {}

    /// Kalman fitter state
    struct state {

        /// State constructor
        ///
        /// @param track_states the vector of track states
        TRACCC_HOST_DEVICE
        explicit state(
            const typename edm::track_fit_collection<
                algebra_type>::device::proxy_type& track,
            const typename edm::track_state_collection<algebra_type>::device&
                track_states,
            const measurement_collection_types::const_device& measurements,
            vecmem::data::vector_view<detray::geometry::barcode>
                sequence_buffer)
            : m_fit_actor_state{track, track_states, measurements},
              m_sequencer_state(
                  vecmem::device_vector<detray::geometry::barcode>(
                      sequence_buffer)),
              m_fit_res{track},
              m_sequence_buffer(sequence_buffer) {}

        /// @return the actor chain state
        TRACCC_HOST_DEVICE
        typename forward_actor_chain_type::state_ref_tuple operator()() {
            return detray::tie(m_aborter_state, m_interactor_state,
                               m_fit_actor_state, m_sequencer_state,
                               m_step_aborter_state);
        }

        /// @return the actor chain state
        TRACCC_HOST_DEVICE
        typename backward_actor_chain_type::state_ref_tuple
        backward_actor_state() {
            return detray::tie(m_aborter_state, m_fit_actor_state,
                               m_interactor_state, m_step_aborter_state);
        }

        /// Individual actor states
        typename aborter::state m_aborter_state{};
        typename interactor::state m_interactor_state{};
        typename forward_fit_actor::state m_fit_actor_state;
        typename barcode_sequencer::state m_sequencer_state;
        kalman_step_aborter::state m_step_aborter_state{};

        /// Fitting result per track
        typename edm::track_fit_collection<algebra_type>::device::proxy_type
            m_fit_res;

        /// View object for barcode sequence
        vecmem::data::vector_view<detray::geometry::barcode> m_sequence_buffer;
    };

    /// Run the kalman fitter for a given number of iterations
    ///
    /// @tparam seed_parameters_t the type of seed track parameter
    ///
    /// @param seed_params seed track parameter
    /// @param fitter_state the state of kalman fitter
    template <typename seed_parameters_t>
    [[nodiscard]] TRACCC_HOST_DEVICE kalman_fitter_status
    fit(const seed_parameters_t& seed_params, state& fitter_state) const {
        seed_parameters_t params = seed_params;
        fitter_state.m_fit_actor_state.reset();

        // Run the kalman filtering for a given number of iterations
        for (std::size_t i = 0; i < m_cfg.n_iterations; i++) {
            if (kalman_fitter_status res = fit_iteration(params, fitter_state);
                res != kalman_fitter_status::SUCCESS) {
                return res;
            }

            // TODO: For multiple iterations, seed parameter should be set to
            // the first track state which has either filtered or smoothed
            // state. If the first track state is a hole, we need to back
            // extrapolate from the filtered or smoothed state of next valid
            // track state.
            params =
                fitter_state.m_fit_actor_state.m_track_states
                    .at(fitter_state.m_fit_actor_state.m_track.state_indices()
                            .at(0))
                    .filtered_params();
            // Reset the iterator of kalman actor
            fitter_state.m_fit_actor_state.reset();
        }

        return kalman_fitter_status::SUCCESS;
    }

    template <typename seed_parameters_t>
    [[nodiscard]] TRACCC_HOST_DEVICE kalman_fitter_status
    fit_iteration(seed_parameters_t params, state& fitter_state) const {
        inflate_covariance(params, m_cfg.covariance_inflation_factor);

        if (kalman_fitter_status res = filter(params, fitter_state);
            res != kalman_fitter_status::SUCCESS) {
            return res;
        }

        // Run smoothing
        if (kalman_fitter_status res = smooth(fitter_state);
            res != kalman_fitter_status::SUCCESS) {
            return res;
        }

        // Update track fitting qualities
        update_statistics(fitter_state);

        check_fitting_result(fitter_state);

        return kalman_fitter_status::SUCCESS;
    }

    /// Run the kalman fitter for an iteration
    ///
    /// @tparam seed_parameters_t the type of seed track parameter
    ///
    /// @param seed_params seed track parameter
    /// @param fitter_state the state of kalman fitter
    template <typename seed_parameters_t>
    [[nodiscard]] TRACCC_HOST_DEVICE kalman_fitter_status
    filter(const seed_parameters_t& seed_params, state& fitter_state) const {

        // Create propagator
        forward_propagator_type propagator(m_cfg.propagation);

        // Set path limit
        fitter_state.m_aborter_state.set_path_limit(
            m_cfg.propagation.stepping.path_limit);

        // Create propagator state
        typename forward_propagator_type::state propagation(
            seed_params, m_field, m_detector, m_cfg.propagation.context);
        propagation.set_particle(detail::correct_particle_hypothesis(
            m_cfg.ptc_hypothesis, seed_params));

        // Set overstep tolerance, stepper constraint and mask tolerance
        propagation._stepping
            .template set_constraint<detray::step::constraint::e_accuracy>(
                m_cfg.propagation.stepping.step_constraint);

        // Reset fitter statistics
        fitter_state.m_fit_res.reset_quality();

        // Run forward filtering
        propagator.propagate(propagation, fitter_state());

        return kalman_fitter_status::SUCCESS;
    }

    /// Run smoothing after kalman filtering
    ///
    /// @brief The smoother is based on "Application of Kalman filtering to
    /// track and vertex fitting", R.Frühwirth, NIM A.
    ///
    /// @param fitter_state the state of kalman fitter
    [[nodiscard]] TRACCC_HOST_DEVICE kalman_fitter_status
    smooth(state& fitter_state) const {

        if (fitter_state.m_sequencer_state.overflow) {
            return kalman_fitter_status::ERROR_BARCODE_SEQUENCE_OVERFLOW;
        }

        fitter_state.m_fit_actor_state.backward_mode = true;
        fitter_state.m_fit_actor_state.reset();

        // Since the smoothed track parameter of the last surface can be
        // considered to be the filtered one, we can reversly iterate the
        // algorithm to obtain the smoothed parameter of other surfaces
        while (!fitter_state.m_fit_actor_state.is_complete() &&
               (!fitter_state.m_fit_actor_state.is_state() ||
                fitter_state.m_fit_actor_state().is_hole())) {
            fitter_state.m_fit_actor_state.next();
        }

        if (fitter_state.m_fit_actor_state.is_complete()) {
            return kalman_fitter_status::SUCCESS;
        }

        auto last = fitter_state.m_fit_actor_state();

        const scalar theta = last.filtered_params().theta();
        if (theta <= 0.f || theta >= constant<traccc::scalar>::pi) {
            return kalman_fitter_status::ERROR_THETA_ZERO;
        }

        if (!std::isfinite(last.filtered_params().phi())) {
            return kalman_fitter_status::ERROR_INVERSION;
        }

        last.smoothed_params().set_parameter_vector(last.filtered_params());
        last.smoothed_params().set_covariance(
            last.filtered_params().covariance());
        last.smoothed_chi2() = last.filtered_chi2();

        if (fitter_state.m_sequencer_state._sequence.empty()) {
            return kalman_fitter_status::SUCCESS;
        }

        // Backward propagator for the two-filters method
        detray::propagation::config backward_cfg = m_cfg.propagation;
        backward_cfg.navigation.min_mask_tolerance =
            static_cast<float>(m_cfg.backward_filter_mask_tolerance);
        backward_cfg.navigation.max_mask_tolerance =
            static_cast<float>(m_cfg.backward_filter_mask_tolerance);

        backward_propagator_type propagator(backward_cfg);

        // Set path limit
        fitter_state.m_aborter_state.set_path_limit(
            m_cfg.propagation.stepping.path_limit);

        typename backward_propagator_type::state propagation(
            last.smoothed_params(), m_field, m_detector,
            fitter_state.m_sequence_buffer, backward_cfg.context);
        propagation.set_particle(detail::correct_particle_hypothesis(
            m_cfg.ptc_hypothesis, last.smoothed_params()));

        assert(std::signbit(
                   propagation._stepping.particle_hypothesis().charge()) ==
               std::signbit(propagation._stepping.bound_params().qop()));

        inflate_covariance(propagation._stepping.bound_params(),
                           m_cfg.covariance_inflation_factor);

        propagation._navigation.set_direction(
            detray::navigation::direction::e_backward);

        // Synchronize the current barcode with the input track parameter
        while (propagation._navigation.get_target_barcode() !=
               last.smoothed_params().surface_link()) {
            assert(!propagation._navigation.is_complete());
            propagation._navigation.next();
        }

        propagator.propagate(propagation, fitter_state.backward_actor_state());

        // Reset the backward mode to false
        fitter_state.m_fit_actor_state.backward_mode = false;

        return kalman_fitter_status::SUCCESS;
    }

    TRACCC_HOST_DEVICE
    void update_statistics(state& fitter_state) const {
        auto& fit_res = fitter_state.m_fit_res;
        auto& track_states = fitter_state.m_fit_actor_state.m_track_states;

        // Fit parameter = smoothed track parameter of the first smoothed track
        // state
        for (unsigned int i : fit_res.state_indices()) {
            if (track_states.at(i).is_smoothed()) {
                fit_res.params() = track_states.at(i).smoothed_params();
                break;
            }
        }

        for (unsigned int i : fit_res.state_indices()) {

            auto trk_state = track_states.at(i);
            const detray::tracking_surface sf{
                m_detector, fitter_state.m_fit_actor_state.m_measurements
                                .at(trk_state.measurement_index())
                                .surface_link};
            statistics_updater<algebra_type>{}(
                fit_res, trk_state,
                fitter_state.m_fit_actor_state.m_measurements);
        }

        // Subtract the NDoF with the degree of freedom of the bound track (=5)
        fit_res.ndf() -= 5.f;
        fit_res.pval() = prob(fit_res.chi2(), fit_res.ndf());

        // The number of holes
        fit_res.nholes() = fitter_state.m_fit_actor_state.n_holes;
    }

    TRACCC_HOST_DEVICE
    void check_fitting_result(state& fitter_state) const {
        auto& fit_res = fitter_state.m_fit_res;
        const auto& track_states =
            fitter_state.m_fit_actor_state.m_track_states;

        // NDF should always be positive for fitting
        if (fit_res.ndf() > 0) {
            for (unsigned int i : fit_res.state_indices()) {
                auto trk_state = track_states.at(i);
                // Fitting fails if any of non-hole track states is not smoothed
                if (!trk_state.is_hole() && !trk_state.is_smoothed()) {
                    fit_res.fit_outcome() =
                        track_fit_outcome::FAILURE_NOT_ALL_SMOOTHED;
                    return;
                }
            }

            // Fitting succeeds if any of non-hole track states is not smoothed
            fit_res.fit_outcome() = track_fit_outcome::SUCCESS;
            return;
        }

        fit_res.fit_outcome() = track_fit_outcome::FAILURE_NON_POSITIVE_NDF;
        return;
    }

    TRACCC_HOST_DEVICE
    const config_type& config() const { return m_cfg; }

    private:
    // Detector object
    const detector_type& m_detector;
    // Field object
    const bfield_type m_field;

    // Configuration object
    config_type m_cfg;
};

}  // namespace traccc
