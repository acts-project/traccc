/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Projection include(s).
#include "traccc/definitions/qualifiers.hpp"

// Detray include(s).
#include "detray/propagator/base_actor.hpp"
#include "detray/propagator/base_stepper.hpp"

namespace traccc {

template <typename interactor_t>
struct interaction_register : detray::actor {
    struct state {
        typename interactor_t::state &interactor_state;
    };

    template <typename propagator_state_t>
    DETRAY_HOST_DEVICE void operator()(state &s,
                                       propagator_state_t &prop_state) const {

        auto &navigation = prop_state._navigation;

        // Do not apply material interaction on the sensitive surface - it is
        // applied outside the propagator
        if (navigation.is_on_sensitive()) {
            s.interactor_state.do_energy_loss = false;
            s.interactor_state.do_multiple_scattering = false;
            s.interactor_state.do_covariance_transport = false;
        } else if (!navigation.is_on_sensitive() && navigation.is_on_module()) {
            s.interactor_state.do_energy_loss = true;
            s.interactor_state.do_multiple_scattering = true;
            s.interactor_state.do_covariance_transport = true;
        }
    }
};

}  // namespace traccc
