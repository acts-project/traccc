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
#include "traccc/utils/particle.hpp"

// detray include(s).
#include <detray/propagator/propagation_config.hpp>

namespace traccc {

/// Configuration struct for track finding
struct finding_config {
    /// Maxmimum number of branches per seed
    unsigned int max_num_branches_per_seed = 10;

    /// Maximum number of branches per surface
    unsigned int max_num_branches_per_surface = 2;

    /// Min/Max number of track candidates per track
    unsigned int min_track_candidates_per_track = 3;
    unsigned int max_track_candidates_per_track = 100;

    /// Min number of track candidates of a specific dimensionality
    unsigned int min_1d_track_candidates_per_track = 0u;
    unsigned int min_2d_track_candidates_per_track = 3u;

    /// Enable strict ordering of 1D measurements after 2D measurements, i.e.
    /// if this is set to true, the algorithm will assume that if it finds a
    /// 1D measurement, it can never again find another 2D measurement.
    bool strict_1d_after_2d_ordering = true;

    /// Maximum allowed number of skipped steps per candidate
    unsigned int max_num_skipping_per_cand = 3;

    /// Minimum step length that track should make to reach the next surface. It
    /// should be set higher than the overstep tolerance not to make it stay on
    /// the same surface
    float min_step_length_for_next_surface = 1.2f * traccc::unit<float>::mm;
    /// Maximum step counts that track can make to reach the next surface
    unsigned int max_step_counts_for_next_surface = 100;

    /// Maximum Chi-square that is allowed for branching
    float chi2_max = 10.f;

    /// Propagation configuration
    detray::propagation::config propagation{};

    /// Minimum momentum for reconstructed tracks
    bool is_min_pT = true;
    float min_p_mag = 100.f * traccc::unit<float>::MeV;

    /// Particle hypothesis
    traccc::pdg_particle<traccc::scalar> ptc_hypothesis =
        traccc::muon<traccc::scalar>();

    /// @name Performance parameters
    /// These parameters impact only compute performance; any case in which a
    /// change in these parameters effects a change in _physics_ performance
    /// should be considered a bug.
    /// @{
    /// @brief The number of links to reserve space for, per seed.
    ///
    /// This parameter describes the number of links which we reserve per seed.
    /// If this number turns out to be too low, the track finding algorithm
    /// will automatically resize it, but this comes at the cost of
    /// performance.
    ///
    /// @note This parameter affects GPU-based track finding only.
    unsigned int initial_links_per_seed = 100;
    /// @}

    /// Set the momentum limit to @param p
    TRACCC_HOST_DEVICE
    inline void min_p(const float p) {
        is_min_pT = false;
        min_p_mag = p;
    }

    /// Set the transverse momentum limit to @param p
    TRACCC_HOST_DEVICE
    inline void min_pT(const float p) {
        is_min_pT = true;
        min_p_mag = p;
    }
};

}  // namespace traccc
