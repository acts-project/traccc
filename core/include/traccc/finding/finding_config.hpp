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

/// Configuration struct for track finding
template <typename scalar_t>
struct finding_config {
    /// @NOTE: This paramter might be removed
    unsigned int max_num_branches_per_seed = 10;

    /// Maximum number of branches per surface
    unsigned int max_num_branches_per_surface = 10;

    /// Min/Max number of track candidates per track
    unsigned int min_track_candidates_per_track = 2;
    unsigned int max_track_candidates_per_track = 30;

    /// Minimum step length that track should make to reach the next surface. It
    /// should be set higher than the overstep tolerance not to make it stay on
    /// the same surface
    scalar_t min_step_length_for_surface_aborter =
        0.1f * detray::unit<scalar_t>::mm;
    /// Maximum Chi-square that is allowed for branching
    scalar_t chi2_max = 15.f;

    /// Constrained step size for propagation
    /// @TODO: Make a separate file for propagation config?
    scalar_t constrained_step_size = std::numeric_limits<scalar_t>::max();

    /// GPU-specific parameter to evaluate the number of measurements to be
    /// iterated per track
    unsigned int n_avg_threads_per_track = 4;
};

}  // namespace traccc