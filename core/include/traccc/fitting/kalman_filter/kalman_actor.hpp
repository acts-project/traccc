/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/kalman_filter/gain_matrix_updater.hpp"
#include "traccc/fitting/kalman_filter/two_filters_smoother.hpp"
#include "traccc/fitting/status_codes.hpp"
#include "traccc/utils/particle.hpp"

// detray include(s).
#include <detray/propagator/base_actor.hpp>

// vecmem include(s)
#include <vecmem/containers/device_vector.hpp>

namespace traccc {

/// Detray actor for Kalman filtering
template <typename algebra_t>
struct kalman_actor : detray::actor {

    // Type declarations
    using track_state_coll = vecmem::device_vector<track_state<algebra_t>>;

    // Actor state
    struct state {

        /// Constructor with the vector of track states
        TRACCC_HOST_DEVICE
        explicit state(track_state_coll track_states)
            : m_track_states(track_states) {
            m_it = m_track_states.begin();
            m_it_rev = m_track_states.rbegin();
        }

        /// @return the reference of track state pointed by the iterator
        TRACCC_HOST_DEVICE
        typename track_state_coll::value_type& operator()() {
            if (!backward_mode) {
                return *m_it;
            } else {
                return *m_it_rev;
            }
        }

        /// Reset the iterator
        TRACCC_HOST_DEVICE
        void reset() {
            m_it = m_track_states.begin();
            m_it_rev = m_track_states.rbegin();
        }

        /// Advance the iterator
        TRACCC_HOST_DEVICE
        void next() {
            if (!backward_mode) {
                m_it++;
            } else {
                m_it_rev++;
            }
        }

        /// @return true if the iterator reaches the end of vector
        TRACCC_HOST_DEVICE
        bool is_complete() {
            if (!backward_mode && m_it == m_track_states.end()) {
                return true;
            } else if (backward_mode && m_it_rev == m_track_states.rend()) {
                return true;
            }
            return false;
        }

        // vector of track states
        track_state_coll m_track_states;

        // iterator for forward filtering
        typename track_state_coll::iterator m_it;

        // iterator for backward filtering
        typename track_state_coll::reverse_iterator m_it_rev;

        // The number of holes (The number of sensitive surfaces which do not
        // have a measurement for the track pattern)
        unsigned int n_holes{0u};

        // Run back filtering for smoothing, if true
        bool backward_mode = false;
    };

    /// Actor operation to perform the Kalman filtering
    ///
    /// @param actor_state the actor state
    /// @param propagation the propagator state
    template <typename propagator_state_t>
    TRACCC_HOST_DEVICE void operator()(state& actor_state,
                                       propagator_state_t& propagation) const {

        auto& stepping = propagation._stepping;
        auto& navigation = propagation._navigation;

        // If the iterator reaches the end, terminate the propagation
        if (actor_state.is_complete()) {
            propagation._heartbeat &= navigation.abort();
            return;
        }

        // triggered only for sensitive surfaces
        if (navigation.is_on_sensitive()) {

            auto& trk_state = actor_state();

            // Increase the hole counts if the propagator fails to find the next
            // measurement
            if (navigation.barcode() != trk_state.surface_link()) {
                if (!actor_state.backward_mode) {
                    actor_state.n_holes++;
                }
                return;
            }

            // This track state is not a hole
            if (!actor_state.backward_mode) {
                trk_state.is_hole = false;
            }

            // Run Kalman Gain Updater
            const auto sf = navigation.get_surface();

            kalman_fitter_status res = kalman_fitter_status::SUCCESS;

            if (!actor_state.backward_mode) {
                // Forward filter
                res = sf.template visit_mask<gain_matrix_updater<algebra_t>>(
                    trk_state, propagation._stepping.bound_params());

                // Update the propagation flow
                stepping.bound_params() = trk_state.filtered();

                // Set full jacobian
                trk_state.jacobian() = stepping.full_jacobian();
            } else {
                // Backward filter for smoothing
                res = sf.template visit_mask<two_filters_smoother<algebra_t>>(
                    trk_state, propagation._stepping.bound_params());
            }

            // Abort if the Kalman update fails
            if (res != kalman_fitter_status::SUCCESS) {
                propagation._heartbeat &= navigation.abort();
                return;
            }

            // Change the charge of hypothesized particles when the sign of qop
            // is changed (This rarely happens when qop is set with a poor seed
            // resolution)
            propagation.set_particle(detail::correct_particle_hypothesis(
                stepping.particle_hypothesis(),
                propagation._stepping.bound_params()));

            // Update iterator
            actor_state.next();

            // Flag renavigation of the current candidate
            navigation.set_high_trust();
        }
    }
};

}  // namespace traccc
