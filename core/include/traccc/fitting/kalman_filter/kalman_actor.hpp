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
#include <detray/navigation/navigator.hpp>
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
        const typename edm::track_fit_collection<algebra_t>::device::proxy_type&
            track,
        const typename edm::track_state_collection<algebra_t>::device&
            track_states,
        const measurement_collection_types::const_device& measurements)
        : m_track{track},
          m_track_states{track_states},
          m_measurements{measurements} {

        reset();
    }

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
    bool is_complete() {
        if (!backward_mode && m_idx == m_track.state_indices().size()) {
            return true;
        } else if (backward_mode && m_idx > m_track.state_indices().size()) {
            return true;
        }
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

    // The number of holes (The number of sensitive surfaces which do not
    // have a measurement for the track pattern)
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

        // If the iterator reaches the end, terminate the propagation
        if (actor_state.is_complete()) {
            propagation._heartbeat &= navigation.exit();
            return;
        }

        // triggered only for sensitive surfaces
        if (navigation.is_on_sensitive()) {

            typename edm::track_state_collection<algebra_t>::device::proxy_type
                trk_state = actor_state();

            // Did the navigation switch direction?
            actor_state.backward_mode =
                navigation.direction() ==
                detray::navigation::direction::e_backward;

            // Increase the hole counts if the propagator fails to find the next
            // measurement

           /* if (navigation.barcode() !=
                actor_state.m_measurements.at(trk_state.measurement_index())
                    .surface_link) {
                if (!actor_state.backward_mode) {
                    actor_state.n_holes++;
                }
                return;
            }

            // This track state is not a hole
            if (!actor_state.backward_mode) {

                trk_state.set_hole(false);
            }*/

            // Increase the hole counts if the propagator fails to find the next
            // measurement
            if (navigation.barcode() != actor_state.m_measurements.at(trk_state.measurement_index())) {
                int i = 1;
                bool found{false};

                // std::cout << "expected " << trk_state.surface_link() << ",
                // found " << navigation.barcode() << std::endl;

                // std::cout << "Track states" << std::endl;
                // The last state could not be found: Abort
                // for (const auto& trk : actor_state.m_track_states) {
                //    std::cout << trk.surface_link() << std::endl;
                //}

                // If the current navigation position can be found at a later
                // track state, then the current track state was skipped:
                // Advance the iterator to keep up with the navigation
                if (actor_state.backward_mode) {
                    // The last state could not be found: Abort
                    if (actor_state.m_it_rev + 1 ==
                        actor_state.m_track_states.rend()) {
                        // Did the navigator find an additional surface?
                        if (trk_state.surface_link() !=
                            std::as_const(navigation)
                                .target()
                                .sf_desc.barcode()) {
                            if (!std::as_const(navigation)
                                     .target()
                                     .sf_desc.is_sensitive()) {
                                return;
                            }
                            trk_state.is_hole = true;
                            actor_state.n_holes++;
                            actor_state.m_it_rev++;  // < prevent double
                                                     // counting of last hole
                            propagation._heartbeat &= navigation.stop();
                            // std::cout << "HOLE last" << std::endl;
                            return;
                        }
                    }
                    // Check how many track states were skipped
                    for (auto itr = actor_state.m_it_rev + 1;
                         itr != actor_state.m_track_states.rend(); ++itr) {
                        if (itr->surface_link() == navigation.barcode()) {
                            // std::cout << "FOUND IT" << std::endl;
                            actor_state.m_it_rev += i;
                            // Only count holes on the most precise fit
                            actor_state.n_holes += i;
                            found = true;
                            for (int j = 0; j < i; ++j) {
                                (itr + j)->is_hole = true;
                                // std::cout << "HOLE skipped" << std::endl;
                            }
                            break;
                        }
                        ++i;
                    }
                } else {
                    // Did the navigator find an additional surface?
                    if (actor_state.m_it + 1 ==
                        actor_state.m_track_states.end()) {
                        if (trk_state.surface_link() !=
                            std::as_const(navigation)
                                .target()
                                .sf_desc.barcode()) {
                            if (!std::as_const(navigation)
                                     .target()
                                     .sf_desc.is_sensitive()) {
                                return;
                            }
                            trk_state.is_hole = true;
                            actor_state.n_holes++;
                            propagation._heartbeat &= navigation.stop();
                            // std::cout << "HOLE last forward" << std::endl;
                            return;
                        }
                    }
                    for (auto itr = actor_state.m_it + 1;
                         itr != actor_state.m_track_states.end(); ++itr) {
                        if (itr->surface_link() == navigation.barcode()) {
                            // std::cout << "FOUND IT" << std::endl;
                            actor_state.m_it += i;
                            found = true;
                            for (int j = 0; j < i; ++j) {
                                // std::cout << "HOLE " << std::endl;
                                (itr + j)->is_hole = true;
                            }
                            break;
                        }
                        ++i;
                    }
                }
                // Navigator found an additional surface: skip
                if (!found) {
                    // std::cout << "test next ";
                    if (actor_state.backward_mode) {
                        // std::cout << actor_state.m_it_rev->surface_link() <<
                        // std::endl;
                    } else {
                        // std::cout << actor_state.m_it->surface_link() <<
                        // std::endl;
                    }
                    return;
                }

            }

            trk_state = actor_state();

            // This track state is not a hole
            trk_state.is_hole = false;

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
                    res = two_filters_smoother<algebra_t>{}(
                        trk_state, actor_state.m_measurements,
                        propagation._stepping.bound_params(), is_line);
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
