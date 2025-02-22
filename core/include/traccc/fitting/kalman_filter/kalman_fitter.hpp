/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/fitting_config.hpp"
#include "traccc/fitting/kalman_filter/gain_matrix_smoother.hpp"
#include "traccc/fitting/kalman_filter/kalman_actor.hpp"
#include "traccc/fitting/kalman_filter/kalman_step_aborter.hpp"
#include "traccc/fitting/kalman_filter/statistics_updater.hpp"
#include "traccc/fitting/kalman_filter/two_filters_smoother.hpp"
#include "traccc/fitting/status_codes.hpp"
#include "traccc/utils/particle.hpp"

// detray include(s).
#include <detray/propagator/actors.hpp>
#include <detray/propagator/propagator.hpp>

// vecmem include(s)
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
    using fit_actor = traccc::kalman_actor<algebra_type>;
    using resetter = detray::parameter_resetter<algebra_type>;

    using actor_chain_type =
        detray::actor_chain<aborter, transporter, interactor, fit_actor,
                            resetter, kalman_step_aborter>;

    using backward_actor_chain_type =
        detray::actor_chain<aborter, transporter, fit_actor, interactor,
                            resetter, kalman_step_aborter>;

    // Propagator type
    using propagator_type =
        detray::propagator<stepper_t, navigator_t, actor_chain_type>;

    using backward_propagator_type =
        detray::propagator<stepper_t, navigator_t, backward_actor_chain_type>;

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
            vecmem::data::vector_view<track_state<algebra_type>> track_states)
            : m_fit_actor_state(
                  vecmem::device_vector<track_state<algebra_type>>(
                      track_states)) {}

        /// State constructor
        ///
        /// @param track_states the vector of track states
        TRACCC_HOST_DEVICE
        explicit state(const vecmem::device_vector<track_state<algebra_type>>&
                           track_states)
            : m_fit_actor_state(track_states) {}

        /// @return the actor chain state
        TRACCC_HOST_DEVICE
        typename actor_chain_type::state operator()() {
            return detray::tie(m_aborter_state, m_transporter_state,
                               m_interactor_state, m_fit_actor_state,
                               m_resetter_state, m_step_aborter_state);
        }

        /// @return the actor chain state
        TRACCC_HOST_DEVICE
        typename backward_actor_chain_type::state backward_actor_state() {
            return detray::tie(m_aborter_state, m_transporter_state,
                               m_fit_actor_state, m_interactor_state,
                               m_resetter_state, m_step_aborter_state);
        }

        /// Individual actor states
        typename aborter::state m_aborter_state{};
        typename transporter::state m_transporter_state{};
        typename interactor::state m_interactor_state{};
        typename fit_actor::state m_fit_actor_state;
        typename resetter::state m_resetter_state{};
        kalman_step_aborter::state m_step_aborter_state{};

        /// Fitting result per track
        fitting_result<algebra_type> m_fit_res;
    };

    /// Run the kalman fitter for a given number of iterations
    ///
    /// @tparam seed_parameters_t the type of seed track parameter
    ///
    /// @param seed_params seed track parameter
    /// @param fitter_state the state of kalman fitter
    template <typename seed_parameters_t>
    TRACCC_HOST_DEVICE [[nodiscard]] kalman_fitter_status fit(
        const seed_parameters_t& seed_params, state& fitter_state) {

        // Run the kalman filtering for a given number of iterations
        for (std::size_t i = 0; i < m_cfg.n_iterations; i++) {

            // Reset the iterator of kalman actor
            fitter_state.m_fit_actor_state.reset();

            auto seed_params_cpy =
                (i == 0) ? seed_params
                         : fitter_state.m_fit_actor_state.m_track_states[0]
                               .smoothed();

            inflate_covariance(seed_params_cpy,
                               m_cfg.covariance_inflation_factor);

            if (kalman_fitter_status res =
                    filter(seed_params_cpy, fitter_state);
                res != kalman_fitter_status::SUCCESS)
                [[unlikely]] { return res; }
        }

        return kalman_fitter_status::SUCCESS;
    }

    /// Run the kalman fitter for an iteration
    ///
    /// @tparam seed_parameters_t the type of seed track parameter
    ///
    /// @param seed_params seed track parameter
    /// @param fitter_state the state of kalman fitter
    template <typename seed_parameters_t>
    TRACCC_HOST_DEVICE [[nodiscard]] kalman_fitter_status filter(
        const seed_parameters_t& seed_params, state& fitter_state) {

        // Create propagator
        propagator_type propagator(m_cfg.propagation);

        // Set path limit
        fitter_state.m_aborter_state.set_path_limit(
            m_cfg.propagation.stepping.path_limit);

        // Create propagator state
        typename propagator_type::state propagation(seed_params, m_field,
                                                    m_detector);
        propagation.set_particle(detail::correct_particle_hypothesis(
            m_cfg.ptc_hypothesis, seed_params));

        // @TODO: Should be removed once detray is fixed to set the
        // volume in the constructor
        propagation._navigation.set_volume(seed_params.surface_link().volume());

        // Set overstep tolerance, stepper constraint and mask tolerance
        propagation._stepping
            .template set_constraint<detray::step::constraint::e_accuracy>(
                m_cfg.propagation.stepping.step_constraint);

        // Reset fitter statistics
        fitter_state.m_fit_res.trk_quality.reset_quality();

        // Run forward filtering
        propagator.propagate(propagation, fitter_state());

        // Run smoothing
        if (kalman_fitter_status res = smooth(fitter_state);
            res != kalman_fitter_status::SUCCESS)
            [[unlikely]] { return res; }

        if (fitter_state.m_fit_res.fit_params.theta() == 0.f)
            [[unlikely]] { return kalman_fitter_status::ERROR_THETA_ZERO; }

        // Update track fitting qualities
        update_statistics(fitter_state);

        return kalman_fitter_status::SUCCESS;
    }

    /// Run smoothing after kalman filtering
    ///
    /// @brief The smoother is based on "Application of Kalman filtering to
    /// track and vertex fitting", R.Frühwirth, NIM A.
    ///
    /// @param fitter_state the state of kalman fitter
    TRACCC_HOST_DEVICE [[nodiscard]] kalman_fitter_status smooth(
        state& fitter_state) {

        auto& track_states = fitter_state.m_fit_actor_state.m_track_states;

        // Since the smoothed track parameter of the last surface can be
        // considered to be the filtered one, we can reversly iterate the
        // algorithm to obtain the smoothed parameter of other surfaces
        for (auto it = track_states.rbegin(); it != track_states.rend(); ++it) {
            if (!(*it).is_hole) {
                fitter_state.m_fit_actor_state.m_it_rev = it;
                break;
            }
            // TODO: Return false because there is no valid track state
            // return false;
        }
        auto& last = *fitter_state.m_fit_actor_state.m_it_rev;
        last.smoothed().set_parameter_vector(last.filtered());
        last.smoothed().set_covariance(last.filtered().covariance());
        last.smoothed_chi2() = last.filtered_chi2();

        if (m_cfg.use_backward_filter) {
            // Backward propagator for the two-filters method
            backward_propagator_type propagator(m_cfg.propagation);

            // Set path limit
            fitter_state.m_aborter_state.set_path_limit(
                m_cfg.propagation.stepping.path_limit);

            typename backward_propagator_type::state propagation(
                last.smoothed(), m_field, m_detector);

            inflate_covariance(propagation._stepping.bound_params(),
                               m_cfg.covariance_inflation_factor);

            propagation._navigation.set_volume(
                last.smoothed().surface_link().volume());

            propagation._navigation.set_direction(
                detray::navigation::direction::e_backward);
            fitter_state.m_fit_actor_state.backward_mode = true;

            const auto& dir = propagation._stepping().dir();
            if (dir[0] == 0.f && dir[1] == 0.f)
                [[unlikely]] {
                    // Particle is exactly parallel to the beampipe, which we
                    // cannot represent.
                    return kalman_fitter_status::ERROR_THETA_ZERO;
                }

            propagator.propagate(propagation,
                                 fitter_state.backward_actor_state());

            // Reset the backward mode to false
            fitter_state.m_fit_actor_state.backward_mode = false;

        } else {
            // Run the Rauch–Tung–Striebel (RTS) smoother
            for (typename vecmem::device_vector<
                     track_state<algebra_type>>::reverse_iterator it =
                     track_states.rbegin() + 1;
                 it != track_states.rend(); ++it) {

                const detray::tracking_surface sf{m_detector,
                                                  it->surface_link()};
                if (kalman_fitter_status res =
                        sf.template visit_mask<
                            gain_matrix_smoother<algebra_type>>(*it, *(it - 1));
                    res != kalman_fitter_status::SUCCESS)
                    [[unlikely]] { return res; }
            }
        }

        return kalman_fitter_status::SUCCESS;
    }

    TRACCC_HOST_DEVICE
    void update_statistics(state& fitter_state) {
        auto& fit_res = fitter_state.m_fit_res;
        auto& track_states = fitter_state.m_fit_actor_state.m_track_states;

        // Fit parameter = smoothed track parameter at the first surface
        fit_res.fit_params = track_states[0].smoothed();

        for (const auto& trk_state : track_states) {

            const detray::tracking_surface sf{m_detector,
                                              trk_state.surface_link()};
            sf.template visit_mask<statistics_updater<algebra_type>>(
                fit_res, trk_state, m_cfg.use_backward_filter);
        }

        // Track quality
        auto& trk_quality = fit_res.trk_quality;

        // Subtract the NDoF with the degree of freedom of the bound track (=5)
        trk_quality.ndf = trk_quality.ndf - 5.f;

        // The number of holes
        trk_quality.n_holes = fitter_state.m_fit_actor_state.n_holes;
    }

    private:
    // Detector object
    const detector_type& m_detector;
    // Field object
    const bfield_type m_field;

    // Configuration object
    config_type m_cfg;
};

}  // namespace traccc
