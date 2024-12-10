/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/definitions/track_parametrization.hpp"
#include "detray/geometry/tracking_surface.hpp"
#include "detray/propagator/base_actor.hpp"
#include "detray/propagator/detail/jacobian_engine.hpp"

namespace detray {

template <typename algebra_t>
struct parameter_resetter : actor {

    template <typename propagator_state_t>
    DETRAY_HOST_DEVICE void operator()(propagator_state_t& propagation) const {

        const auto& navigation = propagation._navigation;
        auto& stepping = propagation._stepping;

        // Do covariance transport when the track is on surface
        if (!(navigation.is_on_sensitive() ||
              navigation.encountered_sf_material())) {
            return;
        }

        // Update free params after bound params were changed by actors
        const auto sf = navigation.get_surface();
        stepping() = sf.bound_to_free_vector(propagation._context,
                                             stepping.bound_params());

        // Reset jacobian transport to identity matrix
        stepping.reset_transport_jacobian();
    }
};

}  // namespace detray
