/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
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

namespace traccc::device {

/// Kalman fitter algorithm to fit a single track
template <typename stepper_t, typename navigator_t>
class sync_kalman_fitter final
    : public kalman_fitter_base<kalman_fitter, stepper_t, navigator_t> {

    public:
    using base_type = kalman_fitter_base<kalman_fitter, stepper_t, navigator_t>;
    using base_type::base_type;

    // vector type
    template <typename T>
    using vector_type = typename base_type::template vector_type<T>;

    // navigator candidate type
    using intersection_type = typename base_type::intersection_type;

    // transform3 type
    using transform3_type = typename base_type::transform3_type;

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

    /// Run the kalman fitter for an iteration
    ///
    /// @tparam seed_parameters_t the type of seed track parameter
    ///
    /// @param seed_params seed track parameter
    /// @param fitter_state the state of kalman fitter
    template <typename seed_parameters_t>
    TRACCC_HOST_DEVICE void filter(
        const seed_parameters_t& seed_params, state& fitter_state,
        vector_type<intersection_type>&& nav_candidates = {}) {}

    TRACCC_HOST_DEVICE void propagate(
        typename propagator_type::state& propagation,
        typename actor_chain_type::state&& actor_states);
};

}  // namespace traccc::device