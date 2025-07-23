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

        // Reset the hole states from a previous fitter pass
        /*for (auto& trk_state : m_track_states) {
            trk_state.is_hole = true;
        }*/
        // n_holes = 0u;
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
    unsigned int count_holes() const {
        unsigned int n_holes{0u};

        for (const auto& trk_state : m_track_states) {
            if (trk_state.is_hole) {
                ++n_holes;
            }
        }

        return n_holes;
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
            if (backward_mode) {

                // for (auto& trk_state2 : m_track_states) {
                // std::cout  << "CONSTR " << trk_state2.surface_link() << ",
                // hole: " << std::boolalpha << trk_state2.is_hole << std::endl;

                //}
                // std::cout  << "In KF: Matched " << trk_state.surface_link()
                // << std::endl;
            }
            trk_state.is_hole = false;
            if (backward_mode) {
                // Continue with KF
                // std::cout  << "Next candidate: " << m_it_rev->surface_link()
                // <<
                // std::endl;
            }
            return true;
        } else {

            if (backward_mode) {
                // std::cout  << navigation.barcode() << "\nvs.\n" <<
                // trk_state.surface_link() << std::endl;
            }
        }

        // If the current navigation surface can be found at a later
        // track state, then the current track state was skipped:
        // Advance the iterator to keep up with the navigation
        int i{1};
        const auto next_sf{std::as_const(navigation).target().sf_desc};
        if (backward_mode) {
            // std::cout  << "Next: " << next_sf.barcode() << std::endl;
        }
        // Additional surfaces: no holes

        // Test if an additional surface was found in the previous step, where
        // the navigation target was not the next surface yet
        if (backward_mode) {
            if (m_it_rev != m_track_states.rbegin()) {
                // std::cout  << "In correct additional " << std::endl;
                for (auto itr = m_it_rev - 1; itr != m_track_states.rbegin();
                     --itr) {
                    // std::cout  << "Correcting additional " <<
                    // itr->surface_link() << std::endl;
                    if (itr->surface_link() == navigation.barcode()) {
                        for (int j = 1; j <= i; ++j) {
                            // std::cout  << "was not hole "  << (m_it_rev -
                            // j)->surface_link() << std::endl;
                            (m_it_rev - j)->is_hole = false;
                        }
                        m_it_rev -= i;
                        // std::cout  << "Next candidate: " <<
                        // m_it_rev->surface_link() << std::endl;
                        return true;
                    }
                    ++i;
                }
            } else if (m_it_rev->surface_link() == navigation.barcode()) {
                // std::cout  << "Last state was not hole" << std::endl;
                m_it_rev->is_hole = false;
                return true;
            }
        } else {
            if (m_it != m_track_states.begin()) {
                // std::cout << "In correct additional " << std::endl;
                for (auto itr = m_it - 1;
                     detray::ranges::distance(m_track_states.begin(), itr) >= 0;
                     --itr) {
                    // std::cout << "Correcting additional " <<
                    // itr->surface_link() << std::endl;
                    if (itr->surface_link() == navigation.barcode()) {
                        for (int j = 1; j <= i; ++j) {
                            // std::cout << "was not hole "  << (m_it -
                            // j)->surface_link() << std::endl;
                            (m_it - j)->is_hole = false;
                        }
                        m_it -= i;
                        // n_holes -= i;
                        // std::cout << "Next candidate: " <<
                        // m_it->surface_link() << std::endl;
                        return true;
                    }
                    ++i;
                }
            }
        }

        // Navigation found additional surface - not a hole
        if (trk_state.surface_link() == next_sf.barcode()) {

            // std::cout << "Additional sf "<< std::endl;
            // std::cout << "Next candidate: " << m_it->surface_link() <<
            // std::endl;
            return false;
        }

        // Skipped surfaces: flag hole(s)
        i = 1;
        if (backward_mode) {
            if (m_it_rev + 1 == m_track_states.rend()) {
                ////std::cout  << "Testing last" << std::endl;
                if (trk_state.surface_link() != next_sf.barcode()) {
                    // std::cout  << "is hole: Could not find last state: " <<
                    // trk_state.surface_link() << std::endl;
                    trk_state.is_hole = true;
                }
                // std::cout  << "Next candidate: " << m_it_rev->surface_link()
                // << std::endl;
                return false;
            }
            for (auto itr = m_it_rev + 1; itr != m_track_states.rend(); ++itr) {

                // std::cout  << "Testing: " << itr->surface_link()<< std::endl;
                if (itr->surface_link() == navigation.barcode()) {

                    // std::cout  << "Found" << std::endl;
                    for (int j = 0; j < i; ++j) {
                        // std::cout  << "is hole" <<(m_it_rev +
                        // j)->surface_link()
                        //<< std::endl;
                        (m_it_rev + j)->is_hole = true;
                    }
                    m_it_rev += i;
                    // The matching candidate is not a hole
                    m_it_rev->is_hole = false;

                    // std::cout  << "Next candidate: " <<
                    // m_it_rev->surface_link()
                    //<< std::endl;
                    return true;
                }
                ++i;
            }
            // If the next surface is a portal/passive, the next sensitive
            // might still be correct
            if (next_sf.is_sensitive()) {
                // std::cout  << "is hole " << trk_state.surface_link() <<
                // std::endl;
                trk_state.is_hole = true;
                ++m_it_rev;
            } else {
                // std::cout  << "Next sf may be in next volume. staying on " <<
                // m_it_rev->surface_link() << std::endl;
            }
        } else {
            if (m_it + 1 == m_track_states.end()) {
                // //std::cout  << "Testing last" << std::endl;
                if (trk_state.surface_link() != next_sf.barcode() &&
                    detray::detail::is_invalid_value(
                        std::as_const(navigation).target().volume_link)) {
                    // std::cout << "is hole: Could not find last state: " <<
                    // trk_state.surface_link() << std::endl;
                    trk_state.is_hole = true;
                }
                // std::cout << "Next candidate: " << m_it->surface_link() <<
                // std::endl;
                return false;
            }
            for (auto itr = m_it + 1; itr != m_track_states.end(); ++itr) {

                // std::cout << "Testing: " << itr->surface_link()<< std::endl;
                if (itr->surface_link() == navigation.barcode()) {

                    // std::cout << "Found" << std::endl;
                    for (int j = 0; j < i; ++j) {
                        // std::cout << "is hole" <<(m_it + j)->surface_link()
                        // << std::endl;
                        (m_it + j)->is_hole = true;
                    }
                    m_it += i;
                    // The matching candidate is not a hole
                    m_it->is_hole = false;

                    // std::cout << "Next candidate: " << m_it->surface_link()
                    // << std::endl;
                    return true;
                }
                ++i;
            }
            // If the next surface is a portal/passive, the next sensitive
            // might still be correct
            if (next_sf.is_sensitive()) {
                // std::cout << "is hole " << trk_state.surface_link() <<
                // std::endl;
                trk_state.is_hole = true;
                ++m_it;
            } else {
                // std::cout << "Next sf may be in next volume. staying on " <<
                // m_it->surface_link() << std::endl;
            }
        }
        // Default case
        // std::cout << "Next candidate: " << m_it->surface_link() << std::endl;
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
    // unsigned int n_holes{0u};

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
            // Did the navigation switch direction?
            actor_state.backward_mode =
                navigation.direction() ==
                detray::navigation::direction::e_backward;

            if (actor_state.is_init()) {
                // std::cout  << "RESET" << std::endl;
                actor_state.reset();
            }

            // Increase the hole counts if the propagator fails to find the next
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
                        // trk_state.is_hole = true;
                        actor_state.next();
                        return;
                    } else {
                        // Backward filter for smoothing
                        res = two_filters_smoother<algebra_t>{}(
                            trk_state, actor_state.m_measurements, bound_param, is_line);
                    }
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
