/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// traccc include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"

// detray include(s).
#include <detray/definitions/pdg_particle.hpp>
#include <detray/propagator/propagation_config.hpp>

namespace traccc {

/// Configuration struct for track finding
struct finding_config {
    /// Maxmimum number of branches per seed
    unsigned int max_num_branches_per_seed = 10;

    /// Maximum number of branches per surface
    unsigned int max_num_branches_per_surface = 10;

    /// Min/Max number of track candidates per track
    unsigned int min_track_candidates_per_track = 3;
    unsigned int max_track_candidates_per_track = 100;

    /// Maximum allowed number of skipped steps per candidate
    unsigned int max_num_skipping_per_cand = 3;

    /// Minimum step length that track should make to reach the next surface. It
    /// should be set higher than the overstep tolerance not to make it stay on
    /// the same surface
    float min_step_length_for_next_surface = 0.5f * traccc::unit<float>::mm;
    /// Maximum step counts that track can make to reach the next surface
    unsigned int max_step_counts_for_next_surface = 100;

    /// Maximum Chi-square that is allowed for branching
    float chi2_max = 10.f;

    /// Propagation configuration
    detray::propagation::config propagation{};

    /// Particle hypothesis
    detray::pdg_particle<traccc::scalar> ptc_hypothesis =
        detray::muon<traccc::scalar>();
};

}  // namespace traccc
