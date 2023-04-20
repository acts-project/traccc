/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/kalman_filter/gain_matrix_smoother.hpp"
#include "traccc/fitting/kalman_filter/gain_matrix_updater.hpp"

// detray include(s).
#include "detray/propagator/base_actor.hpp"

namespace traccc {

/// Detray actor for Kalman filtering
template <typename algebra_t, template <typename...> class vector_t>
struct kalman_actor : detray::actor {

    // Type declarations
    using track_state_type = track_state<algebra_t>;

    // Actor state
    struct state {

        /// Constructor with the vector of track states
        TRACCC_HOST_DEVICE
        state(vector_t<track_state_type>&& track_states)
            : m_track_states(std::move(track_states)) {
            m_it = m_track_states.begin();
        }

        /// Constructor with the vector of track states
        TRACCC_HOST_DEVICE
        state(const vector_t<track_state_type>& track_states)
            : m_track_states(track_states) {
            m_it = m_track_states.begin();
        }

        /// @return the reference of track state pointed by the iterator
        TRACCC_HOST_DEVICE
        track_state_type& operator()() { return *m_it; }

        /// Reset the iterator
        TRACCC_HOST_DEVICE
        void reset() { m_it = m_track_states.begin(); }

        /// Advance the iterator
        TRACCC_HOST_DEVICE
        void next() { m_it++; }

        /// @return true if the iterator reaches the end of vector
        TRACCC_HOST_DEVICE
        bool is_complete() const {
            if (m_it == m_track_states.end()) {
                return true;
            }
            return false;
        }

        // vector of track states
        vector_t<track_state_type> m_track_states;

        // iterator for forward filtering
        typename vector_t<track_state_type>::iterator m_it;
    };

    /// Actor operation to perform the Kalman filtering
    ///
    /// @param actor_state the actor state
    /// @param propagation the propagator state
    template <typename propagator_state_t>
    TRACCC_HOST_DEVICE void operator()(state& actor_state,
                                       propagator_state_t& propagation) const {

        const auto& stepping = propagation._stepping;
        auto& navigation = propagation._navigation;

        // If the iterator reaches the end, terminate the propagation
        if (actor_state.is_complete()) {
            propagation._heartbeat &= navigation.abort();
        }

        // triggered only for sensitive surfaces
        if (navigation.is_on_sensitive()) {

            auto& trk_state = actor_state();

            // Abort if the propagator fails to find the next measurement
            if (navigation.current_object() != trk_state.surface_link()) {
                propagation._heartbeat &= navigation.abort();
            }

            // Set full jacobian
            trk_state.jacobian() = stepping._full_jacobian;

            auto det = navigation.detector();
            const auto& mask_store = det->mask_store();

            // Surface
            const auto& surface = det->surfaces(trk_state.surface_link());

            // Run kalman updater
            mask_store.template visit<gain_matrix_updater<algebra_t>>(
                surface.mask(), trk_state, propagation._stepping._bound_params);

            // Update iterator
            actor_state.next();
        }
    }
};

}  // namespace traccc