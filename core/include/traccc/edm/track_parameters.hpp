/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// traccc include
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/math.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/definitions/track_parametrization.hpp"
#include "traccc/edm/container.hpp"

// detray include(s).
#include "detray/tracks/tracks.hpp"

namespace traccc {

using free_track_parameters =
    detray::free_track_parameters<traccc::default_algebra>;
using bound_track_parameters =
    detray::bound_track_parameters<traccc::default_algebra>;
using free_vector = free_track_parameters::vector_type;
using bound_vector = bound_track_parameters::vector_type;
using bound_covariance = bound_track_parameters::covariance_type;

/// Declare all track_parameters collection types
using bound_track_parameters_collection_types =
    collection_types<bound_track_parameters>;

// Wrap the phi of track parameters to [-pi,pi]
TRACCC_HOST_DEVICE
inline void wrap_phi(bound_track_parameters& param) {

    traccc::scalar phi = param.phi();
    static constexpr traccc::scalar TWOPI =
        2.f * traccc::constant<traccc::scalar>::pi;
    phi = math::fmod(phi, TWOPI);
    if (phi > traccc::constant<traccc::scalar>::pi) {
        phi -= TWOPI;
    } else if (phi < -traccc::constant<traccc::scalar>::pi) {
        phi += TWOPI;
    }
    param.set_phi(phi);
}

}  // namespace traccc
