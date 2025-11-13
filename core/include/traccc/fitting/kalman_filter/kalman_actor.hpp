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
        assert(m_track.constituent_links().at(i).type ==
               edm::track_constituent_link::track_state);
        return m_track_states.at(m_track.constituent_links().at(i).index);
    }

    /// Get the track state at a given position along the track
    TRACCC_HOST_DEVICE
    auto at(unsigned int i) const {
        assert(m_track.constituent_links().at(i).type ==
               edm::track_constituent_link::track_state);
        return m_track_states.at(m_track.constituent_links().at(i).index);
    }

    /// @return the reference of track state pointed by the iterator
    TRACCC_HOST_DEVICE
    typename edm::track_state_collection<algebra_t>::device::proxy_type
    operator()() {
        assert(m_idx >= 0);
        return at(static_cast<unsigned int>(m_idx));
    }

    /// Reset the iterator
    TRACCC_HOST_DEVICE
    void reset() {
        if (!backward_mode) {
            m_idx = 0;
        } else {
            m_idx = static_cast<int>(size()) - 1;
        }
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

    /// @TODO: Const-correctness broken due to a vecmem bug
    /// @returns the number of track states
    TRACCC_HOST_DEVICE
    unsigned int size() /*const*/ { return m_track.constituent_links().size(); }

    /// @return true if the iterator reaches the end of vector
    TRACCC_HOST_DEVICE
    bool is_complete() /*const*/ {
        return (!backward_mode && m_idx == static_cast<int>(size())) ||
               (backward_mode && m_idx == -1);
    }

    /// @TODO: Const-correctness broken due to a vecmem bug
    TRACCC_HOST_DEVICE
    bool is_state() /* const*/ {
        assert(m_idx >= 0);
        return (m_track.constituent_links()
                    .at(static_cast<unsigned int>(m_idx))
                    .type == edm::track_constituent_link::track_state);
    }

    /// @TODO: Const-correctness broken due to a vecmem bug
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

    /// @TODO: Const-correctness broken due to a vecmem bug
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

    /// Object describing the track fit
    typename edm::track_collection<algebra_t>::device::proxy_type m_track;
    /// All track states in the event
    typename edm::track_state_collection<algebra_t>::device m_track_states;
    /// All measurements in the event
    typename edm::measurement_collection<algebra_t>::const_device
        m_measurements;

    /// Index of the current track state
    int m_idx;

    /// The number of holes (The number of sensitive surfaces which do not
    /// have a measurement for the track pattern)
    unsigned int n_holes{0u};

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

        // If the iterator reaches the end, terminate the propagation
        if (actor_state.is_complete()) {
            propagation._heartbeat &= navigation.exit();
            return;
        }

        TRACCC_VERBOSE_HOST_DEVICE("In Kalman actor...");
        TRACCC_VERBOSE_HOST(
            "Expected: " << actor_state().filtered_params().surface_link());

        // triggered only for sensitive surfaces
        if (navigation.is_on_sensitive()) {

            TRACCC_DEBUG_HOST("-> on surface: " << navigation.get_surface());

            typename edm::track_state_collection<algebra_t>::device::proxy_type
                trk_state = actor_state();

            // Increase the hole counts if the propagator fails to find the next
            // measurement
            if (navigation.barcode() !=
                actor_state.m_measurements.at(trk_state.measurement_index())
                    .surface_link()) {
                if (!actor_state.backward_mode) {
                    actor_state.n_holes++;
                }
                return;
            }

            auto& bound_param = stepping.bound_params();

            // Run Kalman Gain Updater
            const auto sf = navigation.get_surface();
            const bool is_line = detail::is_line(sf);

            if (!actor_state.backward_mode) {
                if constexpr (direction_e ==
                                  kalman_actor_direction::FORWARD_ONLY ||
                              direction_e ==
                                  kalman_actor_direction::BIDIRECTIONAL) {
                    // Wrap the phi and theta angles in their valid ranges
                    normalize_angles(bound_param);

                    // Forward filter
                    TRACCC_DEBUG_HOST_DEVICE("Run filtering...");
                    actor_state.fit_result = gain_matrix_updater<algebra_t>{}(
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
                    TRACCC_DEBUG_HOST_DEVICE("Run smoothing...");

                    // Forward filter did not find this state: cannot smoothe
                    if (trk_state.filtered_params().is_invalid()) {
                        TRACCC_ERROR_HOST_DEVICE(
                            "Track state not filtered by forward fit. "
                            "Skipping");
                        actor_state.fit_result =
                            kalman_fitter_status::ERROR_UPDATER_SKIPPED_STATE;
                    } else {
                        actor_state.fit_result =
                            two_filters_smoother<algebra_t>{}(
                                trk_state, actor_state.m_measurements,
                                bound_param, is_line);
                    }
                } else {
                    assert(false);
                }
            }

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
                propagation._heartbeat &=
                    navigation.abort(fitter_debug_msg{actor_state.fit_result});
                return;
            }

            // Change the charge of hypothesized particles when the sign of qop
            // is changed (This rarely happens when qop is set with a poor seed
            // resolution)
            propagation.set_particle(detail::correct_particle_hypothesis(
                stepping.particle_hypothesis(), bound_param));

            // Update iterator
            actor_state.next();

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
