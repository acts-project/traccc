/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/track_collection.hpp"
#include "traccc/edm/track_state_collection.hpp"
#include "traccc/fitting/kalman_filter/gain_matrix_updater.hpp"
#include "traccc/fitting/kalman_filter/is_line_visitor.hpp"
#include "traccc/fitting/kalman_filter/two_filters_smoother.hpp"
#include "traccc/fitting/status_codes.hpp"
#include "traccc/utils/logging.hpp"
#include "traccc/utils/particle.hpp"

// detray include(s).
#include <detray/definitions/navigation.hpp>
#include <detray/propagator/base_actor.hpp>

// vecmem include(s)
#include <vecmem/containers/device_vector.hpp>

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
        const typename edm::track_collection<algebra_t>::device::proxy_type&
            track,
        const typename edm::track_state_collection<algebra_t>::device&
            track_states,
        const typename edm::measurement_collection<algebra_t>::const_device&
            measurements)
        : m_track{track},
          m_track_states{track_states},
          m_measurements{measurements} {

        reset();
    }

    /// Get the track state at a given position along the track
    TRACCC_HOST_DEVICE
    typename edm::track_state_collection<algebra_t>::device::proxy_type at(
        unsigned int i) {
        assert(m_track.constituent_links().at(m_idx).type ==
               edm::track_constituent_link::track_state);
        return m_track_states.at(m_track.constituent_links().at(i).index);
    }

    /// @return the reference of track state pointed by the iterator
    TRACCC_HOST_DEVICE
    typename edm::track_state_collection<algebra_t>::device::proxy_type
    operator()() {
        return at(m_idx);
    }

    /// Advance the iterator
    TRACCC_HOST_DEVICE
    unsigned int size() /*const*/ { return m_track.constituent_links().size(); }

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
        n_holes = 0u;
    }

    /// @return true if the iterator reaches the end of vector
    TRACCC_HOST_DEVICE
    bool finished() /*const*/ {
        return ((!backward_mode && m_idx == size()) ||
                (backward_mode && m_idx > size()));
    }

    /// @returns the current number of missed states during forward fit
    TRACCC_HOST_DEVICE
    unsigned int count_missed_fit() /*const*/ {
        unsigned int n_missed{0u};

        for (unsigned int i = 0u; i < size(); ++i) {
            const auto trk_state = at(i);
            if (!trk_state.is_hole() &&
                trk_state.filtered_params().is_invalid()) {
                TRACCC_DEBUG_HOST_DEVICE(
                    "Missed track state %d/%d on surface %d during forward fit",
                    i, size(), at(i).filtered_params().surface_link().index());
                ++n_missed;
            }
        }

        return n_missed;
    }

    /// @returns the current number of missed states during smoothing
    TRACCC_HOST_DEVICE
    unsigned int count_missed_smoother() /*const*/ {
        unsigned int n_missed{0u};

        for (unsigned int i = 0u; i < size(); ++i) {
            const auto trk_state = at(i);
            if (!trk_state.is_hole() &&
                trk_state.smoothed_params().is_invalid()) {
                TRACCC_DEBUG_HOST_DEVICE(
                    "Missed track state %d/%d on surface %d during smoothing",
                    i, size(), at(i).smoothed_params().surface_link().index());
                ++n_missed;
            }
        }

        return n_missed;
    }

    template <typename nav_state_t>
    TRACCC_HOST_DEVICE bool check_if_hole(const nav_state_t& navigation) {
        if (do_precise_hole_count || !navigation.current().is_edge()) {
            TRACCC_VERBOSE_HOST_DEVICE("---> State flagged as hole");
            ++n_holes;
            return true;
        }
        return false;
    }

    /// @return true if the current surface could be matched to a track state
    /// @TODO: Remove once direct navigator is used in forward pass
    template <typename propagation_state_t>
    TRACCC_HOST_DEVICE bool match_surface_to_track_state(
        propagation_state_t& propagation) {

        const auto& navigation = propagation._navigation;
        auto trk_state = (*this)();

        TRACCC_VERBOSE_HOST("Found: " << navigation.barcode());

        // Surface was found, continue with KF algorithm
        if (navigation.barcode() ==
            trk_state.filtered_params().surface_link()) {
            // Count a hole, if track finding did not find a measurement
            if (!backward_mode && trk_state.is_hole()) {
                TRACCC_VERBOSE_HOST_DEVICE(
                    "-> Track finding flagged this as hole");
                check_if_hole(navigation);
            }

            TRACCC_VERBOSE_HOST_DEVICE(
                "-> Matched current surface to next track state: %d/%d", m_idx,
                size());
            // If track finding did not find measurement on this surface: skip
            return !trk_state.is_hole();
        }

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
                TRACCC_VERBOSE_HOST_DEVICE("--> bw: Evaluate first state");
                check_if_hole(navigation);
                return false;
            }
            TRACCC_VERBOSE_HOST_DEVICE(
                "--> fw: Check other states on track for a match");
            // Check if the current navigation surfaces can be found on a
            // later track state. That means the current track state was
            // skipped by the navigator: Advance the internal iterator
            for (int i = static_cast<int>(m_idx) - 1; i >= 0; --i) {
                if (at(static_cast<unsigned int>(i))
                        .filtered_params()
                        .surface_link() == navigation.barcode()) {
                    TRACCC_VERBOSE_HOST_DEVICE(
                        "--> bw: Matched to earlier state: navigator skipped "
                        "surfaces in between");
                    TRACCC_VERBOSE_HOST_DEVICE("--> bw: no. skipped: %d", n);
                    assert(m_idx >= n);
                    m_idx -= n;
                    assert(std::isfinite(m_idx));
                    fit_result =
                        kalman_fitter_status::ERROR_SMOOTHER_SKIPPED_STATE;
                    return true;
                }
                ++n;
            }
        } else {
            if (m_idx + 1 == size()) {
                TRACCC_DEBUG_HOST_DEVICE("--> fw: Evaluate last state");
                check_if_hole(navigation);
                return false;
            }
            TRACCC_DEBUG_HOST_DEVICE(
                "--> fw: Check other states on track for a match");
            for (unsigned int i = m_idx + 1u; i < size(); ++i) {
                if (at(i).filtered_params().surface_link() ==
                    navigation.barcode()) {
                    TRACCC_DEBUG_HOST_DEVICE(
                        "--> bw: Matched to later state: navigator skipped "
                        "surfaces in between");
                    m_idx += n;
                    fit_result =
                        kalman_fitter_status::ERROR_UPDATER_SKIPPED_STATE;
                    return true;
                }
                ++n;
            }
        }

        // Mismatch was not from missed state: Is a hole
        TRACCC_DEBUG_HOST_DEVICE("--> Did NOT find state: might be hole...");
        const auto is_hole = check_if_hole(navigation);

        if (is_hole) {
            TRACCC_DEBUG_HOST("--> Expected surfaces:");
            for (unsigned int i = 0u; i < size(); ++i) {
                TRACCC_DEBUG_HOST("   - "
                                  << at(i).filtered_params().surface_link());
            }
        } else if (navigation.current().is_edge()) {

            TRACCC_DEBUG_HOST_DEVICE("--> Hit surface edge: Not a hole");
        }

        // After additional surface, keep navigating until match is found
        return false;
    }

    TRACCC_HOST_DEVICE
    bool is_state() {
        return (m_track.constituent_links().at(m_idx).type ==
                edm::track_constituent_link::track_state);
    }

    /// Object describing the track fit
    typename edm::track_collection<algebra_t>::device::proxy_type m_track;
    /// All track states in the event
    typename edm::track_state_collection<algebra_t>::device m_track_states;
    /// All measurements in the event
    typename edm::measurement_collection<algebra_t>::const_device
        m_measurements;

    /// Index of the current track state
    unsigned int m_idx;

    /// The number of holes (The number of sensitive surfaces which do not
    /// have a measurement for the track pattern)
    unsigned int n_holes{0u};

    /// Finish the navigation beyond the track states in the fitter to find all
    /// holes
    bool do_precise_hole_count = false;

    /// Run back filtering for smoothing, if true
    bool backward_mode = false;

    /// Result of the fitter pass
    kalman_fitter_status fit_result = kalman_fitter_status::SUCCESS;
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

        TRACCC_VERBOSE_HOST_DEVICE("In Kalman actor (status %d)...",
                                   actor_state.fit_result);

        // Allow to count holes after the intial track states
        if (actor_state.do_precise_hole_count && actor_state.finished()) {
            if (navigation.is_on_sensitive()) {
                TRACCC_VERBOSE_HOST_DEVICE(
                    "Track is already complete: This surface is a hole");
                // At this point every surface is a hole
                actor_state.n_holes++;
            }
            return;
        }

        TRACCC_VERBOSE_HOST(
            "Expected: " << actor_state().filtered_params().surface_link());

        // triggered only for sensitive surfaces
        if (navigation.is_on_sensitive()) {

            // Did the navigation switch direction?
            actor_state.backward_mode =
                navigation.direction() ==
                detray::navigation::direction::e_backward;

            // Increase the hole count if the propagator stops at an additional
            // surface and wait for the next sensitive surface to match
            if (!actor_state.match_surface_to_track_state(propagation)) {
                return;
            } else if (actor_state.fit_result !=
                       kalman_fitter_status::SUCCESS) {
                // Surface matched but encountered error: Abort fit
                navigation.abort(fitter_debug_msg{actor_state.fit_result});
                propagation._heartbeat = false;
                return;
            }

            // Fetch matched track state
            typename edm::track_state_collection<algebra_t>::device::proxy_type
                trk_state = actor_state();
            auto& bound_param = stepping.bound_params();
            TRACCC_DEBUG_HOST("Before: " << bound_param);

            // Run Kalman Gain Updater
            const auto sf = navigation.current_surface();
            const bool is_line = detail::is_line(sf);

            if (!actor_state.backward_mode) {
                if constexpr (direction_e ==
                                  kalman_actor_direction::FORWARD_ONLY ||
                              direction_e ==
                                  kalman_actor_direction::BIDIRECTIONAL) {
                    // Wrap the phi and theta angles in their valid ranges
                    normalize_angles(propagation._stepping.bound_params());

                    // Forward filter
                    TRACCC_DEBUG_HOST_DEVICE("Run filtering...");
                    actor_state.fit_result = gain_matrix_updater<algebra_t>{}(
                        trk_state, actor_state.m_measurements,
                        propagation._stepping.bound_params(), is_line);

                    // Update the propagation flow
                    stepping.bound_params() = trk_state.filtered_params();
                } else {
                    assert(false);
                }
            } else {
                if constexpr (direction_e ==
                                  kalman_actor_direction::BACKWARD_ONLY ||
                              direction_e ==
                                  kalman_actor_direction::BIDIRECTIONAL) {
                    // Backward filter for smoothing
                    TRACCC_DEBUG_HOST_DEVICE("Run smoothing...");

                    // Forward filter did not find this state: skip
                    if (trk_state.filtered_params().is_invalid()) {
                        TRACCC_ERROR_HOST_DEVICE(
                            "Track state not filtered by forward fit. "
                            "Skipping");
                        actor_state.next();
                        return;
                    }

                    actor_state.fit_result = two_filters_smoother<algebra_t>{}(
                        trk_state, actor_state.m_measurements, bound_param,
                        is_line);

                    TRACCC_DEBUG_HOST(
                        "Smoothed: " << trk_state.smoothed_params());
                } else {
                    assert(false);
                }
            }

            TRACCC_DEBUG_HOST("Filtered: " << bound_param);

            // Abort if the Kalman update fails
            if (actor_state.fit_result != kalman_fitter_status::SUCCESS) {
                if (actor_state.backward_mode) {
                    TRACCC_ERROR_DEVICE("Abort backward fit: KF status %d",
                                        actor_state.fit_result);
                    TRACCC_ERROR_HOST(
                        "Abort backward fit: "
                        << fitter_debug_msg{actor_state.fit_result}());
                } else {
                    TRACCC_ERROR_DEVICE("Abort forward fit: KF status %d",
                                        actor_state.fit_result);
                    TRACCC_ERROR_HOST("Abort forward fit: " << fitter_debug_msg{
                                          actor_state.fit_result}());
                }
                navigation.abort(fitter_debug_msg{actor_state.fit_result});
                propagation._heartbeat = false;
                return;
            }

            // Change the charge of hypothesized particles when the sign of qop
            // is changed (This rarely happens when qop is set with a poor seed
            // resolution)
            propagation.set_particle(detail::correct_particle_hypothesis(
                stepping.particle_hypothesis(), bound_param));

            // Update iterator
            actor_state.next();

            // No need to continue
            if (actor_state.finished() && !actor_state.do_precise_hole_count) {
                navigation.exit();
                propagation._heartbeat = false;
            }

            // Flag renavigation of the current candidate (unless for overlap)
            if (math::fabs(navigation()) > 1.f * unit<float>::um) {
                navigation.set_high_trust();
            } else {
                TRACCC_DEBUG_HOST_DEVICE(
                    "Encountered overlap, jump to next surface");
            }
        }
    }
};

}  // namespace traccc
