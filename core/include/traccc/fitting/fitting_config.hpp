/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// detray include(s).
#include "detray/definitions/pdg_particle.hpp"
#include "detray/definitions/units.hpp"
#include "detray/propagator/propagation_config.hpp"

namespace traccc {

/// Configuration struct for track fitting
struct fitting_config {

    std::size_t n_iterations = 1;

    /// Propagation configuration
    detray::propagation::config propagation{};

    /// Particle hypothesis
    detray::pdg_particle<traccc::scalar> ptc_hypothesis =
        detray::muon<traccc::scalar>();
};

}  // namespace traccc
