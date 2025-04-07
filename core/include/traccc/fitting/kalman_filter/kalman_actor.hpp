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
#include <detray/navigation/navigator.hpp>
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

        /// @return true if the iterator reaches the end of vector
        /// @TODO: Remove once direct navigator is used in forward pass
        template <typename propagation_state_t>
        TRACCC_HOST_DEVICE bool check_matching_surface(
            propagation_state_t& propagation) {

            auto& navigation = propagation._navigation;
            auto& trk_state = (*this)();

            // Surface was found, continue with KF algorithm
            if (navigation.barcode() == trk_state.surface_link()) {
                trk_state.is_hole = false;
                // Continue with KF
                return true;
            }

            // If the current navigation surface can be found at a later
            // track state, then the current track state was skipped:
            // Advance the iterator to keep up with the navigation
            int i{1};
            const auto next_sf{std::as_const(navigation).target().sf_desc};
            if (backward_mode) {
                // The last state could not be found: Abort
                if (m_it_rev + 1 == m_track_states.rend()) {
                    // Check if the current track state is found on the next sf
                    if (trk_state.surface_link() != next_sf.barcode()) {
                        trk_state.is_hole = true;
                        n_holes++;
                        // Try the next surface after this
                        return false;
                    }
                }
                // Check how many track states might have been skipped
                for (auto itr = m_it_rev + 1; itr != m_track_states.rend();
                     ++itr) {
                    if (itr->surface_link() == navigation.barcode()) {
                        m_it_rev += i;
                        // Only count holes in backward mode: most precise fit
                        n_holes += i;
                        for (int j = 0; j < i; ++j) {
                            (itr + j)->is_hole = true;
                        }
                        return true;
                    }
                    ++i;
                }
            } else {
                if (m_it + 1 == m_track_states.end()) {
                    if (trk_state.surface_link() != next_sf.barcode()) {
                        trk_state.is_hole = true;
                        return false;
                    }
                }
                for (auto itr = m_it + 1; itr != m_track_states.end(); ++itr) {
                    if (itr->surface_link() == navigation.barcode()) {
                        m_it += i;
                        for (int j = 0; j < i; ++j) {
                            (itr + j)->is_hole = true;
                        }
                        return true;
                    }
                    ++i;
                }
            }
            // Default case
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
            propagation._heartbeat &= navigation.pause();
            return;
        }

        // triggered only for sensitive surfaces
        if (navigation.is_on_sensitive()) {
            // Did the navigation switch direction?
            actor_state.backward_mode =
                navigation.direction() ==
                detray::navigation::direction::e_backward;

            // Increase the hole counts if the propagator fails to find the next
            // measurement and wait for the next sensitive surface
            if (!actor_state.check_matching_surface(propagation)) {
                return;
            }

            auto& trk_state = actor_state();
            auto& bound_param = stepping.bound_params();

            // Run Kalman Gain Updater
            const auto sf = navigation.get_surface();

            kalman_fitter_status res = kalman_fitter_status::SUCCESS;

            if (!actor_state.backward_mode) {
                res = sf.template visit_mask<gain_matrix_updater<algebra_t>>(
                    trk_state, bound_param);

                // Update the propagation flow
                bound_param = trk_state.filtered();

            } else {
                // Backward filter for smoothing
                res = sf.template visit_mask<two_filters_smoother<algebra_t>>(
                    trk_state, bound_param);
            }

            // Abort if the Kalman update fails
            if (res != kalman_fitter_status::SUCCESS) {
                propagation._heartbeat &= navigation.abort();
                return;
            }

            // Has been flagged as a hole state when navigation stopped at
            // additional surfaces: Revert this
            if (trk_state.is_hole) {
                actor_state.n_holes--;
                trk_state.is_hole = false;
            }

            // Change the charge of hypothesized particles when the sign of qop
            // is changed (This rarely happens when qop is set with a poor seed
            // resolution)
            propagation.set_particle(detail::correct_particle_hypothesis(
                stepping.particle_hypothesis(), bound_param));

            // Update iterator
            actor_state.next();

            // Flag renavigation of the current candidate
            navigation.set_high_trust();
        }
    }
};

}  // namespace traccc
