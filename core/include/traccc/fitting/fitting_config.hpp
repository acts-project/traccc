/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// detray include(s).
#include "detray/definitions/units.hpp"
#include "detray/propagator/propagation_config.hpp"

namespace traccc {

/// Configuration struct for track fitting
template <typename scalar_t>
struct fitting_config {

    std::size_t n_iterations = 1;

    /// Propagation configuration
    detray::propagation::config<scalar_t> propagation{};
};

}  // namespace traccc
