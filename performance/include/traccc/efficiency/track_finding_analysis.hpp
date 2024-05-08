/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/particle.hpp"
#include "traccc/edm/track_candidate.hpp"

namespace traccc::performance {

/// Tool for analyzing the track finding efficiency
class track_finding_analysis {

    public:
    /// Default constructor
    track_finding_analysis();

    /// Analyze the tracks found in a specific event.
    ///
    /// @param reco_particles The reconstructed particles
    /// @param truth_particles The truth particles
    ///
    void analyze(
        const track_candidate_container_types::const_view& reco_particles,
        const particle_container_types::const_view& truth_particles);

    private:
    /// Function checking whether a truth and reconstructed particle match.
    ///
    /// @param reco The reconstructed particle
    /// @param truth The truth particle
    ///
    /// @return Whether the particles match
    ///
    bool match(
        const track_candidate_container_types::const_device::const_element_view&
            reco,
        const particle_container_types::const_device::const_element_view& truth)
        const;

};  // class track_finding_analysis

}  // namespace traccc::performance
