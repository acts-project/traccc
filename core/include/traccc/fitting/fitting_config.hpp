/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// detray include(s).
#include "detray/definitions/units.hpp"

namespace traccc {

/// Configuration struct for track fitting
template <typename scalar_t>
struct fitting_config {

    std::size_t n_iterations = 1;
    scalar_t pathlimit = std::numeric_limits<scalar_t>::max();
    scalar_t overstep_tolerance = -10 * detray::unit<scalar_t>::um;
    scalar_t step_constraint = std::numeric_limits<scalar_t>::max();
};

}  // namespace traccc