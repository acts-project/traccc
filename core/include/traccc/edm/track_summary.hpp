/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/track_parameters.hpp"

namespace traccc {

struct track_summary {

    /// (Mandatory) Seed track parameter
    bound_track_parameters seed;

    /// (Optional) Fitted track parameter at the both ends
    thrust::pair<bound_track_parameters, bound_track_parameters>
        fitted_params_at_tip;

    /// (Optional) Number of degree of freedoms of the track
    scalar ndf{0.f};

    /// (Optional) Chi square from finding/fitting algorithm
    scalar chi2{std::numeric_limits<float>::max()};

    /// (Optional) The number of holes
    unsigned int n_holes{0u};

    // @TODO: make constructors
    // track_summary() = delete;
};

}  // namespace traccc