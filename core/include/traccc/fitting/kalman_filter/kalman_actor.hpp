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
#include <detray/utils/log.hpp>

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

    /// Advance the iterator
    TRACCC_HOST_DEVICE
    typename edm::track_state_collection<algebra_t>::device::proxy_type at(
        unsigned int i) {
        return m_track_states.at(m_track.state_indices().at(i));
    }

    /// Advance the iterator
    TRACCC_HOST_DEVICE
    unsigned int size() /*const*/ { return m_track.state_indices().size(); }

    /// Advance the iterator
    TRACCC_HOST_DEVICE
    void next() {
        if (!backward_mode) {
            m_idx++;
        } else {
            m_idx--;
        }
    }

    /// Reset the iterator
    TRACCC_HOST_DEVICE
    void reset() {
        if (!backward_mode) {
            m_idx = 0;
        } else {
            m_idx = size() - 1;
        }

        // Reset the hole states from a previous fitter pass
        for (unsigned int i = 0u; i < size(); ++i) {
            at(i).set_hole(false);
        }
        n_holes = 0u;
    }

    /// @return true if the iterator reaches the end of vector
    TRACCC_HOST_DEVICE
    bool is_complete() /*const*/ {
        return ((!backward_mode && m_idx == size()) ||
                (backward_mode && m_idx > size()));
    }

    /// @returns the current number of holes in this state
    TRACCC_HOST_DEVICE
    unsigned int count_missed() /*const*/ {
        unsigned int n_missed{0u};

        for (unsigned int i = 0u; i < size(); ++i) {
            if (at(i).filtered_params().is_invalid()) {
                ++n_missed;
            }
        }

        return n_missed;
    }

    template <typename nav_state_t>
    TRACCC_HOST_DEVICE void check_if_hole(const nav_state_t& navigation) {
        if (do_precise_hole_count || !navigation.current().is_edge()) {
            ++n_holes;
        }
    }

    /// @return true if the iterator reaches the end of vector
    /// @TODO: Remove once direct navigator is used in forward pass
    template <typename propagation_state_t>
    TRACCC_HOST_DEVICE bool check_matching_surface(
        propagation_state_t& propagation) {

        auto& navigation = propagation._navigation;
        auto trk_state = (*this)();

        DETRAY_VERBOSE_HOST("Checking: " << navigation.barcode());
        DETRAY_VERBOSE_HOST(
            "Expected: " << trk_state.filtered_params().surface_link());

        // Surface was found, continue with KF algorithm
        if (navigation.barcode() ==
            trk_state.filtered_params().surface_link()) {
            // Count a hole, if track finding did not find a measurement
            if (trk_state.is_hole()) {
                DETRAY_VERBOSE_HOST_DEVICE("state might be flagged hole");
                check_if_hole(navigation);
            }

            DETRAY_VERBOSE_HOST_DEVICE("Matched");
            // If track finding did not find measurement on this surface: skip
            return !trk_state.is_hole();
        }

        DETRAY_VERBOSE_HOST_DEVICE("state not flagged hole");

        // Skipped surfaces: adjust iterator and remove counted hole
        // (only relevant if using non-direct navigation, e.g. forward truth
        // fitting or different prop. config between CKF asnd KF)
        // TODO: Remove again
        // using detector_t = typename propagation_state_t::detector_type;
        // using nav_state_t = typename
        // propagation_state_t::navigator_state_type;
        /*if constexpr (!std::same_as<nav_state_t,
                                    typename detray::direct_navigator<
                                        detector_t>::state>) {*/
        unsigned int n{1};
        if (backward_mode) {
            // If we are on the last state and the navigation surface does
            // not match, it must be an additional surface
            // -> continue navigation until matched
            if (m_idx == 0u) {
                check_if_hole(navigation);
                return false;
            }
            // Check if the current navigation surfaces can be found on a
            // later track state. That means the current track state was
            // skipped by the navigator: Advance the internal iterator
            for (int i = m_idx + 1; i >= 0; --i) {
                if (at(i).filtered_params().surface_link() ==
                    navigation.barcode()) {
                    assert(m_idx >= n);
                    m_idx -= n;
                    return true;
                }
                ++n;
            }
        } else {
            if (m_idx + 1 == size()) {
                DETRAY_VERBOSE_HOST_DEVICE("evaluate last state");
                check_if_hole(navigation);
                return false;
            }
            DETRAY_VERBOSE_HOST_DEVICE("Check other states for match");
            for (unsigned int i = m_idx + 1u; i < size(); ++i) {
                if (at(i).filtered_params().surface_link() ==
                    navigation.barcode()) {
                    DETRAY_VERBOSE_HOST_DEVICE("found state: skipped not hole");
                    m_idx += n;
                    return true;
                }
                ++n;
            }
        }
        //}

        // Mismatch was not from missed state: Is a hole
        DETRAY_VERBOSE_HOST_DEVICE("NOT found state: might be hole");
        check_if_hole(navigation);

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

    // Run back filtering for smoothing, if true
    bool do_precise_hole_count = false;
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
            if (!actor_state.do_precise_hole_count) {
                propagation._heartbeat &= navigation.exit();
            } else if (navigation.is_on_sensitive()) {
                // At this point the surface is always a hole
                actor_state.n_holes++;
            }
            return;
        }

        // triggered only for sensitive surfaces
        if (navigation.is_on_sensitive()) {

            DETRAY_VERBOSE_HOST("\nIn actor: " << navigation.barcode());

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

            const bool is_line = detail::is_line(sf);

            kalman_fitter_status res = kalman_fitter_status::SUCCESS;

            if (!actor_state.backward_mode) {
                if constexpr (direction_e ==
                                  kalman_actor_direction::FORWARD_ONLY ||
                              direction_e ==
                                  kalman_actor_direction::BIDIRECTIONAL) {
                    // Forward filter
                    res = gain_matrix_updater<algebra_t>{}(
                        trk_state, actor_state.m_measurements, bound_param,
                        is_line);

                    // Update the propagation flow
                    bound_param = trk_state.filtered_params();
                } else {
                    assert(false);
                }
            } else {
                if constexpr (direction_e ==
                                  kalman_actor_direction::BACKWARD_ONLY ||
                              direction_e ==
                                  kalman_actor_direction::BIDIRECTIONAL) {

                    // Forward filter did not find this state: skip
                    if (trk_state.filtered_params().is_invalid()) {
                        actor_state.next();
                        return;
                    }
                    // Backward filter for smoothing
                    res = two_filters_smoother<algebra_t>{}(
                        trk_state, actor_state.m_measurements, bound_param,
                        is_line);
                } else {
                    assert(false);
                }
            }

            // Abort if the Kalman update fails
            DETRAY_DEBUG_HOST(
                "KALMAN FITTER STATUS: " << fitter_debug_msg{res}());
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
            if (math::fabs(navigation()) > 1.f * unit<float>::um) {
                navigation.set_high_trust();
            }

            // Need propagation to reach the naext candidate
            /*if (math::fabs(navigation()) > 1.f * unit<float>::um) {
                DETRAY_DEBUG_HOST("DIST: " << math::fabs(navigation()));
                return;
            } else {
                DETRAY_DEBUG_HOST("KALMAN ACTOR OVERLAP");
                // Jump to the next candidate
                const auto free_track =
                    sf.bound_to_free_vector(propagation._context, bound_param);
                using navigator_t = typename propagator_state_t::
                    navigator_state_type::navigator_type;
                constexpr navigator_t navigator{};
                navigator.update(free_track, navigation, {},
                                 propagation._context);
            }*/
        }
    }
};

}  // namespace traccc
