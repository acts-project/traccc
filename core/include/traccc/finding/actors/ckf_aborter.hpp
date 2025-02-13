/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "traccc/definitions/primitives.hpp"

// detray include(s)
#include <detray/definitions/detail/qualifiers.hpp>
#include <detray/propagator/base_actor.hpp>

// System include(s)
#include <limits>

namespace traccc {

/// Aborter triggered when the next surface is reached
struct ckf_aborter : detray::actor {
    struct state {
        // minimal step length to prevent from staying on the same surface
        scalar min_step_length = 0.5f;
        /// Maximum step counts that track can make to reach the next surface
        unsigned int max_count = 100;

        bool success = false;
        unsigned int count = 0;

        scalar path_from_surface{0.f};
    };

    template <typename propagator_state_t>
    DETRAY_HOST_DEVICE void operator()(state &abrt_state,
                                       propagator_state_t &prop_state) const {

        auto &navigation = prop_state._navigation;
        auto &stepping = prop_state._stepping;

        abrt_state.count++;
        abrt_state.path_from_surface += stepping.step_size();

        // Abort at the next sensitive surface
        if (navigation.is_on_sensitive() &&
            abrt_state.path_from_surface > abrt_state.min_step_length) {
            prop_state._heartbeat &= navigation.abort();
            abrt_state.success = true;
        }

        // Reset path from surface
        if (navigation.is_on_sensitive() ||
            navigation.encountered_sf_material()) {
            abrt_state.path_from_surface = 0.f;
        }

        if (abrt_state.count > abrt_state.max_count) {
            prop_state._heartbeat &= navigation.abort();
        }
    }
};

}  // namespace traccc
