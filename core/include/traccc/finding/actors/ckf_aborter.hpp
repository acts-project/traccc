/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/propagator/base_actor.hpp"
#include "detray/propagator/base_stepper.hpp"
#include "traccc/definitions/primitives.hpp"

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
    };

    template <typename propagator_state_t>
    DETRAY_HOST_DEVICE void operator()(state &abrt_state,
                                       propagator_state_t &prop_state) const {

        auto &navigation = prop_state._navigation;
        auto &stepping = prop_state._stepping;

        abrt_state.count++;

        // Abort at the next sensitive surface
        if (navigation.is_on_sensitive() &&
            stepping.path_from_surface() > abrt_state.min_step_length) {
            prop_state._heartbeat &= navigation.abort();
            abrt_state.success = true;
        }

        if (abrt_state.count > abrt_state.max_count) {
            prop_state._heartbeat &= navigation.abort();
        }
    }
};

}  // namespace traccc
