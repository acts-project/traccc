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

// System include(s)
#include <limits>

namespace traccc {

/// Aborter triggered when the next surface is reached
struct surface_id_aborter : detray::actor {
    struct state {
        detray::geometry::barcode m_abort_id;
    };

    template <typename propagator_state_t>
    DETRAY_HOST_DEVICE void operator()(state &abrt_state,
                                       propagator_state_t &prop_state) const {

        auto &navigation = prop_state._navigation;

        // Abort if the propagator is on the surface with abort ID
        if ((navigation.is_on_sensitive() ||
             navigation.encountered_sf_material())) {
            if (navigation.barcode() == abrt_state.m_abort_id) {
                prop_state._heartbeat &= navigation.abort();
            }
        }
    }
};

}  // namespace traccc
