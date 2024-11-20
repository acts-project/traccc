/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/track_parameters.hpp"

// System include(s).
#include <limits>

namespace traccc {

struct fitting_result {

    /// Fitted track parameter at the first track state
    bound_track_parameters fitted_params_initial;

    /// Fitted track parameter at the last track state
    bound_track_parameters fitted_params_final;

    /// Number of degree of freedoms of the track
    scalar ndf{0.f};

    /// Chi square from finding/fitting algorithm
    scalar chi2{std::numeric_limits<float>::max()};

    /// The number of holes
    unsigned int n_holes{0u};
};

}  // namespace traccc