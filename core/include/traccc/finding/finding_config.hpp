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

/// Configuration struct for track finding
template <typename scalar_t>
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
    scalar_t min_step_length_for_next_surface =
        0.5f * detray::unit<scalar_t>::mm;
    /// Maximum step counts that track can make to reach the next surface
    unsigned int max_step_counts_for_next_surface = 100;

    /// Maximum Chi-square that is allowed for branching
    scalar_t chi2_max = 30.f;

    /// Propagation configuration
    detray::propagation::config<scalar_t> propagation{};

    /****************************
     *  GPU-specfic parameters
     ****************************/
    /// The number of measurements to be iterated per thread
    unsigned int n_measurements_per_thread = 8;

    /// Max number of candidates per seed used for navigation buffer creation
    /// @NOTE: This is supposed to be larger than (at least equal to)
    /// max_num_branches_per_seed
    unsigned int navigation_buffer_size_scaler = 20;
};

}  // namespace traccc
