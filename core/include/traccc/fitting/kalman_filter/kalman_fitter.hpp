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
#include "traccc/utils/particle.hpp"

// detray include(s).
#include "detray/propagator/actor_chain.hpp"
#include "detray/propagator/actors/aborters.hpp"
#include "detray/propagator/actors/parameter_resetter.hpp"
#include "detray/propagator/actors/parameter_transporter.hpp"
#include "detray/propagator/actors/pointwise_material_interactor.hpp"
#include "detray/propagator/propagator.hpp"

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

    // vector type
    template <typename T>
    using vector_type = typename detector_type::template vector_type<T>;

    /// Configuration type
    using config_type = fitting_config;

    // Field type
    using bfield_type = typename stepper_t::magnetic_field_type;

    // Actor types
    using aborter = detray::pathlimit_aborter;
    using transporter = detray::parameter_transporter<algebra_type>;
    using interactor = detray::pointwise_material_interactor<algebra_type>;
    using fit_actor = traccc::kalman_actor<algebra_type, vector_type>;
    using resetter = detray::parameter_resetter<algebra_type>;

    using actor_chain_type =
        detray::actor_chain<detray::dtuple, aborter, transporter, interactor,
                            fit_actor, resetter, kalman_step_aborter>;

    // Propagator type
    using propagator_type =
        detray::propagator<stepper_t, navigator_t, actor_chain_type>;

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
        state(vector_type<track_state<algebra_type>>&& track_states)
            : m_fit_actor_state(std::move(track_states)) {}

        /// State constructor
        ///
        /// @param track_states the vector of track states
        TRACCC_HOST_DEVICE
        state(const vector_type<track_state<algebra_type>>& track_states)
            : m_fit_actor_state(track_states) {}

        /// @return the actor chain state
        TRACCC_HOST_DEVICE
        typename actor_chain_type::state operator()() {
            return detray::tie(m_aborter_state, m_transporter_state,
                               m_interactor_state, m_fit_actor_state,
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
    TRACCC_HOST_DEVICE void fit(const seed_parameters_t& seed_params,
                                state& fitter_state) {

        // Run the kalman filtering for a given number of iterations
        for (std::size_t i = 0; i < m_cfg.n_iterations; i++) {

            // Reset the iterator of kalman actor
            fitter_state.m_fit_actor_state.reset();

            if (i == 0) {
                filter(seed_params, fitter_state);
            }
            // From the second iteration, seed parameter is the smoothed track
            // parameter at the first surface
            else {
                const auto& new_seed_params =
                    fitter_state.m_fit_actor_state.m_track_states[0].smoothed();

                filter(new_seed_params, fitter_state);
            }
        }
    }

    /// Run the kalman fitter for an iteration
    ///
    /// @tparam seed_parameters_t the type of seed track parameter
    ///
    /// @param seed_params seed track parameter
    /// @param fitter_state the state of kalman fitter
    template <typename seed_parameters_t>
    TRACCC_HOST_DEVICE void filter(const seed_parameters_t& seed_params,
                                   state& fitter_state) {

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

        // Run forward filtering
        propagator.propagate(propagation, fitter_state());

        // Run smoothing
        smooth(fitter_state);

        // Update track fitting qualities
        update_statistics(fitter_state);
    }

    /// Run smoothing after kalman filtering
    ///
    /// @brief The smoother is based on "Application of Kalman filtering to
    /// track and vertex fitting", R.Frühwirth, NIM A.
    ///
    /// @param fitter_state the state of kalman fitter
    TRACCC_HOST_DEVICE
    void smooth(state& fitter_state) {
        auto& track_states = fitter_state.m_fit_actor_state.m_track_states;

        // The smoothing algorithm requires the following:
        // (1) the filtered track parameter of the current surface
        // (2) the smoothed track parameter of the next surface
        //
        // Since the smoothed track parameter of the last surface can be
        // considered to be the filtered one, we can reversly iterate the
        // algorithm to obtain the smoothed parameter of other surfaces
        auto& last = track_states.back();
        last.smoothed().set_parameter_vector(last.filtered());
        last.smoothed().set_covariance(last.filtered().covariance());
        last.smoothed_chi2() = last.filtered_chi2();

        for (typename vector_type<track_state<algebra_type>>::reverse_iterator
                 it = track_states.rbegin() + 1;
             it != track_states.rend(); ++it) {

            // Run kalman smoother
            const detray::tracking_surface sf{m_detector, it->surface_link()};
            sf.template visit_mask<gain_matrix_smoother<algebra_type>>(
                *it, *(it - 1));
        }
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
            sf.template visit_mask<statistics_updater<algebra_type>>(fit_res,
                                                                     trk_state);
        }

        // Subtract the NDoF with the degree of freedom of the bound track (=5)
        fit_res.ndf = fit_res.ndf - 5.f;

        // The number of holes
        fit_res.n_holes = fitter_state.m_fit_actor_state.n_holes;
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
