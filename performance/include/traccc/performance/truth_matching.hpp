/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/particle.hpp"
#include "traccc/edm/seed.hpp"
#include "traccc/edm/track_candidate.hpp"

namespace traccc::performance {

/// Class providing reco <-> truth track matching
///
/// It can be used either on its own to check the match between reconstructed
/// and truth tracks, or as a base class for classes that do some higher level
/// performance analysis.
///
class truth_matching {

    public:
    /// Configuration for the track matching
    struct config {
        /// Uncertainty in millimeters, withing which measurements would match
        float m_measurement_uncertainty{1.f * unit<float>::mm};
        /// Fraction of reco measurements required to match truth measurements
        float m_measurement_match{1.0};
    };  // struct config

    /// Constructor with a configuration
    truth_matching(const config& cfg);

    /// Function checking whether a truth particle and a track seed match
    ///
    /// @param reco The track seed
    /// @param spacepoints The spacepoint collection that the seeds refer to
    /// @param truth The truth particle
    ///
    /// @return Whether the seed matches the truth particle
    ///
    bool matches(
        const seed& reco,
        const spacepoint_collection_types::const_view spacepoints,
        const particle_container_types::const_device::const_element_view& truth)
        const;

    /// Function checking whether a truth particle and reconstructed track match
    ///
    /// @param reco The reconstructed track
    /// @param truth The truth particle
    ///
    /// @return Whether the reconstructed track matches the truth particle
    ///
    bool matches(
        const track_candidate_container_types::const_device::const_element_view&
            reco,
        const particle_container_types::const_device::const_element_view& truth)
        const;

    /// Function checking whether two measurement objects match
    ///
    /// @param reco The reconstructed measurement
    /// @param truth The truth measurement
    ///
    /// @return Wehther the measurements match
    ///
    bool matches(const measurement& reco, const measurement& truth) const;

    private:
    /// Configuration for the tool
    config m_config;

};  // class truth_matching

}  // namespace traccc::performance
