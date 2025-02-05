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
#include "traccc/utils/particle.hpp"

// detray include(s).
#include <detray/navigation/navigator.hpp>
#include <detray/propagator/base_actor.hpp>

// vecmem include(s)
#include <vecmem/containers/device_vector.hpp>

namespace traccc {

/// Detray actor for Kalman filtering
template <typename algebra_t>
struct kalman_actor : detray::actor {

    // Type declarations
    using track_state_type = track_state<algebra_t>;

    // Define filter direction
    using enum detray::navigation::direction;

    // Actor state
    struct state {
        using iterator_t =
            typename vecmem::device_vector<track_state_type>::iterator;

        /// Constructor with the vector of track states
        TRACCC_HOST_DEVICE
        state(vecmem::data::vector_view<track_state_type> track_states,
              bool backward_mode = false)
            : m_begin{track_states.ptr()},
              m_end{track_states.ptr() + track_states.size()} {
            reset(backward_mode);
        }

        /// Constructor with the vector of track states
        TRACCC_HOST_DEVICE
        state(vecmem::device_vector<track_state<algebra_t>>& track_states,
              bool backward_mode = false)
            : m_begin{track_states.begin()}, m_end{track_states.end()} {
            reset(backward_mode);
        }

        /// Get track state range iterators
        /// @{
        TRACCC_HOST_DEVICE iterator_t begin() const { return m_begin; }
        TRACCC_HOST_DEVICE iterator_t end() const { return m_end; }
        /// @}

        /// @return a reference of the track state pointed to by the iterator
        TRACCC_HOST_DEVICE track_state_type& operator()() { return *m_it; }

        /// Reset the iterator
        TRACCC_HOST_DEVICE
        void reset(bool backward_mode = false) {
            m_it = backward_mode ? end() - 1 : begin();
        }

        // Reset the iterator
        TRACCC_HOST_DEVICE
        void reset(detray::navigation::direction dir) {
            reset(dir == e_backward);
        }

        /// Move the iterator forward
        TRACCC_HOST_DEVICE void next() { ++m_it; }

        /// Move the iterator backward
        TRACCC_HOST_DEVICE void previous() { --m_it; }

        /// @return true if the iterator reaches the end of vector
        TRACCC_HOST_DEVICE
        bool is_complete(bool backward_mode = false) {
            return (!backward_mode && m_it == end()) ||
                   (backward_mode && m_it == begin() - 1);
        }

        /// First track state of the track
        iterator_t m_begin;
        /// Last track state of the track
        iterator_t m_end;
        /// Current track state
        iterator_t m_it;

        // The number of holes (The number of sensitive surfaces which do not
        // have a measurement for the track pattern)
        unsigned int n_holes{0u};
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
        const bool backward_mode{navigation.direction() ==
                                 detray::navigation::direction::e_backward};

        // If the iterator reaches the end, terminate the propagation
        if (actor_state.is_complete(backward_mode)) {
            propagation._heartbeat &= navigation.abort();
            return;
        }

        // triggered only for sensitive surfaces
        if (navigation.is_on_sensitive()) {

            auto& trk_state = actor_state();

            // Increase the hole counts if the propagator fails to find the next
            // measurement
            if (navigation.barcode() != trk_state.surface_link()) {
                if (!backward_mode) {
                    actor_state.n_holes++;
                }
                return;
            }

            // Run Kalman Gain Updater
            const auto sf = navigation.get_surface();

            bool res = false;

            if (!backward_mode) {
                // This track state is not a hole
                trk_state.is_hole = false;

                // Forward filter
                res = sf.template visit_mask<gain_matrix_updater<algebra_t>>(
                    trk_state, propagation._stepping.bound_params());

                // Update the propagation flow
                stepping.bound_params() = trk_state.filtered();

                // Set full jacobian
                trk_state.jacobian() = stepping.full_jacobian();

                // Update iterator
                actor_state.next();
            } else {
                // Backward filter for smoothing
                res = sf.template visit_mask<two_filters_smoother<algebra_t>>(
                    trk_state, propagation._stepping.bound_params());

                // Update iterator
                actor_state.previous();
            }

            // Abort if the Kalman update fails
            if (!res) {
                propagation._heartbeat &= navigation.abort();
                return;
            }

            // Change the charge of hypothesized particles when the sign of qop
            // is changed (This rarely happens when qop is set with a poor seed
            // resolution)
            propagation.set_particle(detail::correct_particle_hypothesis(
                stepping.particle_hypothesis(),
                propagation._stepping.bound_params()));

            // Flag renavigation of the current candidate
            navigation.set_high_trust();
        }
    }
};

}  // namespace traccc
