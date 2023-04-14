/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/kalman_filter/kalman_actor.hpp"

// detray include(s).
#include "detray/propagator/actor_chain.hpp"
#include "detray/propagator/actors/aborters.hpp"
#include "detray/propagator/actors/parameter_resetter.hpp"
#include "detray/propagator/actors/parameter_transporter.hpp"
#include "detray/propagator/actors/pointwise_material_interactor.hpp"
#include "detray/propagator/propagator.hpp"

namespace traccc {

/// Kalman fitter algorithm to fit a single track
template <typename stepper_t, typename navigator_t>
class kalman_fitter {

    public:
    // scalar type
    using scalar_type = typename stepper_t::scalar_type;

    // vector type
    template <typename T>
    using vector_type = typename navigator_t::template vector_type<T>;

    // navigator candidate type
    using intersection_type = typename navigator_t::intersection_type;

    // Kalman fitter configuration
    struct config {
        std::size_t n_iterations = 1;
        scalar_type pathlimit = std::numeric_limits<scalar>::max();
        scalar_type overstep_tolerance = -10 * detray::unit<scalar>::um;
        scalar_type step_constraint = 5. * detray::unit<scalar>::mm;
    };

    // transform3 type
    using transform3_type = typename stepper_t::transform3_type;

    // Detector type
    using detector_type = typename navigator_t::detector_type;

    // Actor types
    using aborter = detray::pathlimit_aborter;
    using transporter = detray::parameter_transporter<transform3_type>;
    using interactor = detray::pointwise_material_interactor<transform3_type>;
    using fit_actor = traccc::kalman_actor<transform3_type, vector_type>;
    using resetter = detray::parameter_resetter<transform3_type>;

    using actor_chain_type =
        detray::actor_chain<std::tuple, aborter, transporter, interactor,
                            fit_actor, resetter>;

    // Propagator type
    using propagator_type =
        detray::propagator<stepper_t, navigator_t, actor_chain_type>;

    /// Constructor with a detector
    ///
    /// @param det the detector object
    TRACCC_HOST_DEVICE
    kalman_fitter(const detector_type& det) : m_detector(det) {}

    /// Kalman fitter state
    struct state {

        /// State constructor
        ///
        /// @param track_states the vector of track states
        TRACCC_HOST_DEVICE
        state(vector_type<track_state<transform3_type>>&& track_states)
            : m_fit_actor_state(std::move(track_states)) {}

        /// State constructor
        ///
        /// @param track_states the vector of track states
        TRACCC_HOST_DEVICE
        state(const vector_type<track_state<transform3_type>>& track_states)
            : m_fit_actor_state(track_states) {}

        /// @return the actor chain state
        TRACCC_HOST_DEVICE
        typename actor_chain_type::state operator()() {
            return std::tie(m_aborter_state, m_transporter_state,
                            m_interactor_state, m_fit_actor_state,
                            m_resetter_state);
        }

        /// Individual actor states
        typename aborter::state m_aborter_state{};
        typename transporter::state m_transporter_state{};
        typename interactor::state m_interactor_state{};
        typename fit_actor::state m_fit_actor_state;
        typename resetter::state m_resetter_state{};

        /// Fitting result per track
        fitter_info<transform3_type> m_fit_info;
    };

    /// Run the kalman fitter for a given number of iterations
    ///
    /// @tparam seed_parameters_t the type of seed track parameter
    ///
    /// @param seed_params seed track parameter
    /// @param fitter_state the state of kalman fitter
    template <typename seed_parameters_t>
    TRACCC_HOST_DEVICE void fit(
        const seed_parameters_t& seed_params, state& fitter_state,
        vector_type<intersection_type>&& nav_candidates = {}) {

        // Run the kalman filtering for a given number of iterations
        for (std::size_t i = 0; i < m_cfg.n_iterations; i++) {

            // Reset the iterator of kalman actor
            fitter_state.m_fit_actor_state.reset();

            if (i == 0) {
                filter(seed_params, fitter_state, std::move(nav_candidates));
            }
            // From the second iteration, seed parameter is the smoothed track
            // parameter at the first surface
            else {
                const auto& new_seed_params =
                    fitter_state.m_fit_actor_state.m_track_states[0].smoothed();

                filter(new_seed_params, fitter_state,
                       std::move(nav_candidates));
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
    TRACCC_HOST_DEVICE void filter(
        const seed_parameters_t& seed_params, state& fitter_state,
        vector_type<intersection_type>&& nav_candidates = {}) {

        // Create propagator
        propagator_type propagator({}, {});

        // Set path limit
        fitter_state.m_aborter_state.set_path_limit(m_cfg.pathlimit);

        // Create propagator state
        typename propagator_type::state propagation(
            seed_params, m_detector.get_bfield(), m_detector,
            std::move(nav_candidates));

        // Set overstep tolerance and stepper constraint
        propagation._stepping().set_overstep_tolerance(
            m_cfg.overstep_tolerance);
        propagation._stepping
            .template set_constraint<detray::step::constraint::e_accuracy>(
                m_cfg.step_constraint);

        // Run forward filtering
        propagator.propagate(propagation, fitter_state());

        // Run smoothing
        smooth(fitter_state);

        //@todo: Write track info
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
        last.smoothed().set_vector(last.filtered().vector());
        last.smoothed().set_covariance(last.filtered().covariance());

        const auto& mask_store = m_detector.mask_store();

        for (typename vector_type<
                 track_state<transform3_type>>::reverse_iterator it =
                 track_states.rbegin() + 1;
             it != track_states.rend(); ++it) {

            // Surface
            const auto& surface = m_detector.surfaces(it->surface_link());

            // Run kalman smoother
            mask_store.template visit<gain_matrix_smoother<transform3_type>>(
                surface.mask(), *it, *(it - 1));
        }
    }

    private:
    // Detector object
    const detector_type& m_detector;
    // Configuration object
    config m_cfg;
};

}  // namespace traccc