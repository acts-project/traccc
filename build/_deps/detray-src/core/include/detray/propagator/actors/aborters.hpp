/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/propagator/base_actor.hpp"
#include "detray/propagator/base_stepper.hpp"

// System include(s)
#include <limits>

namespace detray {

/// Aborter that checks whether the track has exceeded its pathlimit
struct pathlimit_aborter : actor {

    /// Pathlimit for a single propagation workflow
    struct state {
        /// Absolute path limit
        scalar _path_limit = std::numeric_limits<scalar>::max();

        /// Set the path limit to a scalar @param pl
        DETRAY_HOST_DEVICE
        inline void set_path_limit(const scalar pl) { _path_limit = pl; }

        /// @returns this states remaining path length.
        DETRAY_HOST_DEVICE
        inline scalar path_limit() const { return _path_limit; }
    };

    /// Enforces the path limit on a stepper state
    ///
    /// @param abrt_state contains the path limit
    /// @param prop_state state of the propagation
    template <typename propagator_state_t>
    DETRAY_HOST_DEVICE void operator()(state &abrt_state,
                                       propagator_state_t &prop_state) const {
        auto &step_state = prop_state._stepping;
        auto &nav_state = prop_state._navigation;

        // Nothing left to do. Propagation will exit successfully
        if (nav_state.is_complete()) {
            return;
        }

        const scalar step_limit =
            abrt_state.path_limit() -
            math::fabs(prop_state._stepping.abs_path_length());

        // Check the path limit
        if (step_limit <= 0.f) {
            // Stop navigation
            prop_state._heartbeat &= nav_state.abort();
        }

        // Don't go over the path limit in the next step
        step_state.template set_constraint<step::constraint::e_aborter>(
            step_limit);
    }
};

/// Aborter checks whether a specific surface was reached
struct target_aborter : actor {

    /// Keeps the index for the target surface
    struct state {
        /// Unique surface id of the target
        geometry::barcode _target_surface;
    };

    /// Exits the navigation as soon as the target surface has been found.
    ///
    /// @param abrt_state contains the target surface index
    /// @param prop_state state of the propagation
    template <typename propagator_state_t>
    DETRAY_HOST_DEVICE void operator()(const state &abrt_state,
                                       propagator_state_t &prop_state) const {

        auto &navigation = prop_state._navigation;
        const auto &stepping = prop_state._stepping;

        // In case the propagation starts on a module, make sure to not abort
        // directly
        if (navigation.is_on_surface() &&
            (navigation.barcode() == abrt_state._target_surface) &&
            (stepping.path_length() > 0.f)) {
            prop_state._heartbeat &= navigation.abort();
        }
    }
};

}  // namespace detray
