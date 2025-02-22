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
#include <detray/tracks/tracks.hpp>

namespace traccc {

template <detray::concepts::algebra algebra_t = traccc::default_algebra>
using free_track_parameters = detray::free_track_parameters<algebra_t>;

template <detray::concepts::algebra algebra_t = traccc::default_algebra>
using bound_track_parameters = detray::bound_track_parameters<algebra_t>;

template <detray::concepts::algebra algebra_t = traccc::default_algebra>
using free_vector = typename free_track_parameters<algebra_t>::vector_type;

template <detray::concepts::algebra algebra_t = traccc::default_algebra>
using bound_vector = typename bound_track_parameters<algebra_t>::vector_type;

template <detray::concepts::algebra algebra_t = traccc::default_algebra>
using bound_covariance =
    typename bound_track_parameters<algebra_t>::covariance_type;

template <detray::concepts::algebra algebra_t = traccc::default_algebra>
using bound_matrix = detray::bound_matrix<algebra_t>;

/// Declare all track_parameters collection types
using bound_track_parameters_collection_types =
    collection_types<bound_track_parameters<>>;

// Wrap the phi of track parameters to [-pi,pi]
template <detray::concepts::algebra algebra_t>
TRACCC_HOST_DEVICE inline void wrap_phi(
    bound_track_parameters<algebra_t>& param) {

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

/// Covariance inflation used for track fitting
template <detray::concepts::algebra algebra_t>
TRACCC_HOST_DEVICE inline void inflate_covariance(
    bound_track_parameters<algebra_t>& param, const traccc::scalar inf_fac) {
    auto& cov = param.covariance();
    for (unsigned int i = 0; i < e_bound_size; i++) {
        for (unsigned int j = 0; j < e_bound_size; j++) {
            if (i == j) {
                getter::element(cov, i, i) *= inf_fac;
            } else {
                getter::element(cov, i, j) = 0.f;
            }
        }
    }
}

}  // namespace traccc
