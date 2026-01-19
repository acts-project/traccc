/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/track_parameters.hpp"

// System include(s)
#include <limits>

namespace traccc {

/// Data payload that is accumulated during the Kalman track follower
struct track_state_candidate {
    // The index of a matched measurement
    unsigned int measurement_index{std::numeric_limits<unsigned int>::max()};
};

/// Kalman data payload extended by the filtered track parameters
template <detray::concepts::algebra algebra_t>
struct filtered_track_state_candidate : public track_state_candidate {
    // The filtered chi2
    detray::dscalar<algebra_t> filtered_chi2;

    // The filtered track parameters corresponding to the measurement
    traccc::bound_track_parameters<algebra_t> filtered_params{};
};

/// Full Kalman data payload
template <detray::concepts::algebra algebra_t>
struct full_track_state_candidate
    : public filtered_track_state_candidate<algebra_t> {
    // The predicted track parameters at the measurement surface
    traccc::bound_track_parameters<algebra_t> predicted_params{};

    // The full Jacobian from the previous sensitive surface to the current one
    traccc::bound_matrix<algebra_t> jacobian{};
};

}  // namespace traccc
