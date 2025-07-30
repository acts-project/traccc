/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_fit_collection.hpp"
#include "traccc/edm/track_state_collection.hpp"
#include "traccc/fitting/kalman_filter/gain_matrix_updater.hpp"
#include "traccc/fitting/kalman_filter/is_line_visitor.hpp"
#include "traccc/fitting/kalman_filter/two_filters_smoother.hpp"
#include "traccc/fitting/status_codes.hpp"
#include "traccc/utils/particle.hpp"

// detray include(s).
#include <detray/navigation/direct_navigator.hpp>
#include <detray/navigation/navigator.hpp>
#include <detray/propagator/base_actor.hpp>

// vecmem include(s)
#include <vecmem/containers/device_vector.hpp>

// System include(s)
#include <cstdlib>

namespace traccc {

enum class kalman_actor_direction {
    FORWARD_ONLY,
    BACKWARD_ONLY,
    BIDIRECTIONAL
};

template <typename algebra_t>
struct kalman_actor_state {

    /// Constructor with the vector of track states
    TRACCC_HOST_DEVICE
    kalman_actor_state(
        const typename edm::track_fit_collection<algebra_t>::device::proxy_type&
            track,
        const typename edm::track_state_collection<algebra_t>::device&
            track_states,
        const measurement_collection_types::const_device& measurements)
        : m_track{track},
          m_track_states{track_states},
          m_measurements{measurements} {}

    /// @return the reference of track state pointed by the iterator
    TRACCC_HOST_DEVICE
    typename edm::track_state_collection<algebra_t>::device::proxy_type
    operator()() {
        return m_track_states.at(m_track.state_indices().at(m_idx));
    }

    /// Reset the iterator
    TRACCC_HOST_DEVICE
    void reset() {
        if (!backward_mode) {
            m_idx = 0;
        } else {
            m_idx = m_track.state_indices().size() - 1;
        }

        // Reset the hole states from a previous fitter pass
        /*for (auto& trk_state : m_track_states) {
            trk_state.is_hole = true;
        }*/
        n_holes = 0u;
    }

    /// Advance the iterator
    TRACCC_HOST_DEVICE
    void next() {
        if (!backward_mode) {
            m_idx++;
        } else {
            m_idx--;
        }
    }

    /// @return true if the iterator reaches the end of vector
    TRACCC_HOST_DEVICE
    bool is_complete() /*const*/ {
        return ((!backward_mode && m_idx == m_track.state_indices().size()) ||
               (backward_mode && m_idx > m_track.state_indices().size()));
    }

    /// @returns the current number of holes in this state
    TRACCC_HOST_DEVICE
    unsigned int count_missed() const {
        unsigned int n_missed{0u};

        if (backward_mode) {
            for (const auto& trk_state : m_track_states) {
                if (trk_state.smoothed().is_invalid()) {
                    ++n_missed;
                }
            }
        } else {
            for (const auto& trk_state : m_track_states) {
                if (trk_state.filtered().is_invalid()) {
                    ++n_missed;
                }
            }
        }

        return n_missed;
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
            // Count a hole, if track finding did not find a measurement
            if (trk_state.is_hole) {
                ++n_holes;
            }
            // If track finding did not find measurement on this surface: skip
            return !trk_state.is_hole;
        }

        // Skipped surfaces: adjust iterator and remove counted hole
        // (only relevant if using non-direct navigation, e.g. forward truth
        // fitting or different prop. config between CKF asnd KF)
        // TODO: Remove again
        using detector_t = typename propagation_state_t::detector_type;
        using nav_state_t = typename propagation_state_t::navigator_state_type;
        if constexpr (!std::same_as<nav_state_t,
                                    typename detray::direct_navigator<
                                        detector_t>::state>) {
            int i{1};
            if (backward_mode) {
                // If we are on the last state and the navigation surface does
                // not match, it must be an additional surface
                // -> continue navigation until matched
                if (m_it_rev + 1 == m_track_states.rend()) {
                    ++n_holes;
                    return false;
                }
                // Check if the current navigation surfaces can be found on a
                // later track state. That means the current track state was
                // skipped by the navigator: Advance the internal iterator
                for (auto itr = m_it_rev + 1; itr != m_track_states.rend();
                     ++itr) {
                    if (itr->surface_link() == navigation.barcode()) {
                        m_it_rev += i;
                        return true;
                    }
                    ++i;
                }
            } else {
                if (m_it + 1 == m_track_states.end()) {
                    ++n_holes;
                    return false;
                }
                for (auto itr = m_it + 1; itr != m_track_states.end(); ++itr) {
                    if (itr->surface_link() == navigation.barcode()) {
                        m_it += i;
                        return true;
                    }
                    ++i;
                }
            }
        }

        // Mismatch was not from missed state: Is a hole
        ++n_holes;

        // After additional surface, keep navigating until match is found
        return false;
    }

    TRACCC_HOST_DEVICE
    bool is_state() {
        return m_track.state_indices().at(m_idx) !=
               std::numeric_limits<unsigned int>::max();
    }

    /// Object describing the track fit
    typename edm::track_fit_collection<algebra_t>::device::proxy_type m_track;
    /// All track states in the event
    typename edm::track_state_collection<algebra_t>::device m_track_states;
    /// All measurements in the event
    measurement_collection_types::const_device m_measurements;

    /// Index of the current track state
    unsigned int m_idx;

    // Count the number of encountered surfaces without measurement
    unsigned int n_holes{0u};

    // Run back filtering for smoothing, if true
    bool backward_mode = false;
};

/// Detray actor for Kalman filtering
template <typename algebra_t, kalman_actor_direction direction_e>
struct kalman_actor : detray::actor {

    // Actor state
    using state = kalman_actor_state<algebra_t>;

    /// Actor operation to perform the Kalman filtering
    ///
    /// @param actor_state the actor state
    /// @param propagation the propagator state
    template <typename propagator_state_t>
    TRACCC_HOST_DEVICE void operator()(state& actor_state,
                                       propagator_state_t& propagation) const {

        auto& stepping = propagation._stepping;
        auto& navigation = propagation._navigation;

        if (actor_state.is_complete()) {
            propagation._heartbeat &= navigation.exit();
            return;
        }

        // triggered only for sensitive surfaces
        if (navigation.is_on_sensitive()) {

            // Did the navigation switch direction?
            actor_state.backward_mode =
                navigation.direction() ==
                detray::navigation::direction::e_backward;

            // Increase the hole count if the propagator stops at an additional
            // surface and wait for the next sensitive surface to match
            if (!actor_state.check_matching_surface(propagation)) {
                return;
            }

            auto trk_state = actor_state();
            auto& bound_param = stepping.bound_params();

            // Run Kalman Gain Updater
            const auto sf = navigation.get_surface();

            const bool is_line = sf.template visit_mask<is_line_visitor>();

            kalman_fitter_status res = kalman_fitter_status::SUCCESS;

            if (!actor_state.backward_mode) {
                if constexpr (direction_e ==
                                  kalman_actor_direction::FORWARD_ONLY ||
                              direction_e ==
                                  kalman_actor_direction::BIDIRECTIONAL) {
                    // Forward filter
                    res = gain_matrix_updater<algebra_t>{}(
                        trk_state, actor_state.m_measurements,
                        bound_param, is_line);

                    // Update the propagation flow
                    bound_param = trk_state.filtered();
                } else {
                    assert(false);
                }
            } else {
                if constexpr (direction_e ==
                                  kalman_actor_direction::BACKWARD_ONLY ||
                              direction_e ==
                                  kalman_actor_direction::BIDIRECTIONAL) {

                    // Forward filter did not find this state: skip
                    if (trk_state.filtered().is_invalid()) {
                        actor_state.next();
                        return;
                    }
                    // Backward filter for smoothing
                    res = two_filters_smoother<algebra_t>{}(
                        trk_state, actor_state.m_measurements, bound_param, is_line);
                } else {
                    assert(false);
                }
            }

            // Abort if the Kalman update fails
            if (res != kalman_fitter_status::SUCCESS) {
                propagation._heartbeat &=
                    navigation.abort(fitter_debug_msg{res});
                return;
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
