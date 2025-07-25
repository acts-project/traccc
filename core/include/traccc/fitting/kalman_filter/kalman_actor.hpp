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
    using track_state_coll = vecmem::device_vector<track_state<algebra_t>>;

    /// Constructor with the vector of track states
    TRACCC_HOST_DEVICE
    explicit kalman_actor_state(track_state_coll track_states)
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
            // std::cout << "In KF: Matched " << trk_state.surface_link() <<
            // std::endl;
            trk_state.is_hole = false;
            // Continue with KF
            // std::cout << "Next candidate: " << m_it->surface_link() <<
            // std::endl;
            return true;
        } else {
            // std::cout << navigation.barcode() << "\nvs.\n" <<
            // trk_state.surface_link() << std::endl;
        }

        // If the current navigation surface can be found at a later
        // track state, then the current track state was skipped:
        // Advance the iterator to keep up with the navigation
        int i{1};
        const auto next_sf{std::as_const(navigation).target().sf_desc};

        // std::cout << "Next: " << next_sf.barcode() << std::endl;

        // Additional surfaces: no holes

        // Test if an additional surface was found in the previous step, where
        // the navigation target was not the next surface yet
        if (backward_mode) {
            if ((m_it_rev - 1)->surface_link() == navigation.barcode()) {
                --m_it_rev;
                (m_it_rev - 1)->is_hole = false;
                --n_holes;
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
            // The last track state could not be found
            /*if (m_it_rev + 1 == m_track_states.rend()) {
                // Check if the current track state is found on the next sf
                if (trk_state.surface_link() != next_sf.barcode()) {
                    // If the target is a portal, the next sensitive might still
                    // be found correctly
                    if (next_sf.is_sensitive()) {
                        trk_state.is_hole = true;
                        n_holes++;
                    }
                    // Try the next surface after this
                    return false;
                }
            }
            // Check how many track states might have been skipped
            for (auto itr = m_it_rev + 1; itr != m_track_states.rend(); ++itr) {
                if (itr->surface_link() == navigation.barcode()) {
                    for (int j = 0; j < i; ++j) {
                        (m_it_rev + j)->is_hole = true;
                    }
                    m_it_rev += i;
                    // Only count holes in backward mode: most precise fit
                    n_holes += static_cast<unsigned int>(i);
                    return true;
                }
                ++i;
            }

            // If the next surface is a portal/passive, the next sensitive
            // might still be correct
            if (next_sf.is_sensitive()) {
                trk_state.is_hole = true;
                ++m_it_rev;
                n_holes++;
            }*/
        } else {
            if (m_it + 1 == m_track_states.end()) {
                // std::cout << "Testing last" << std::endl;
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

/// Detray actor for Kalman filtering
template <typename algebra_t, kalman_actor_direction direction_e>
struct kalman_actor : detray::actor {

    // Type declarations
    using track_state_coll = vecmem::device_vector<track_state<algebra_t>>;

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

            // Increase the hole counts if the propagator fails to find the next
            // measurement and wait for the next sensitive surface
            if (!actor_state.check_matching_surface(propagation)) {
                return;
            }

            auto& trk_state = actor_state();
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
                        trk_state, bound_param, is_line);

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
                    // Backward filter for smoothing
                    res = two_filters_smoother<algebra_t>{}(
                        trk_state, bound_param, is_line);
                } else {
                    assert(false);
                }
            }

            // Abort if the Kalman update fails
            if (res != kalman_fitter_status::SUCCESS) {
                propagation._heartbeat &=
                    navigation.abort(fitter_debug_msg{res});

                // std::cout << "Rest is holes!" << std::endl;
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
