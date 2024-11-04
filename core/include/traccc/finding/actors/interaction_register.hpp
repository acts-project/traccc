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

struct interaction_register : detray::actor {

    template <typename interactor_state_t, typename propagator_state_t>
    DETRAY_HOST_DEVICE void operator()(
        interactor_state_t &s, const propagator_state_t &prop_state) const {

        const auto &navigation = prop_state._navigation;

        // Do not apply material interaction on the sensitive surface - it is
        // applied outside the propagator
        if (navigation.is_on_sensitive()) {
            s.do_energy_loss = false;
            s.do_multiple_scattering = false;
            s.do_covariance_transport = false;
        } else if (!navigation.is_on_sensitive() &&
                   navigation.encountered_sf_material()) {
            s.do_energy_loss = true;
            s.do_multiple_scattering = true;
            s.do_covariance_transport = true;
        }
    }
};

}  // namespace traccc
