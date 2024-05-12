/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/utils/helpers.hpp"

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/edm/particle.hpp"
#include "traccc/edm/track_candidate.hpp"

// System include(s).
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace traccc::performance {

// Forward declaration(s).
namespace details {
struct track_finding_analysis_data;
}

/// Tool for analyzing the track finding efficiency
class track_finding_analysis {

    public:
    /// Configuration for the efficiency analysis
    struct config {

        /// Name of the output file to write
        std::string m_output_name{"track_finding_efficiency.root"};

        /// Truth particle PDG ID to take into consideration
        std::vector<int> m_truth_pdgid{-13, 13};
        /// Minimum truth pT to take into consideration
        float m_truth_pt_min{500.f * unit<float>::MeV};
        /// Maximum (absolute) truth pseudorapidity to take into consideration
        float m_truth_eta_max{2.8f};

        /// Uncertainty in millimeters, withing which measurements would match
        float m_measurement_uncertainty{1.f * unit<float>::mm};
        /// Fraction of reco measurements required to match truth measurements
        float m_measurement_match{1.0};

        /// Binning for the efficiency vs. eta plot
        plot_helpers::binning m_eta_binning{
            "Track Finding Efficiency vs. #eta;#eta", 40, -3.f, 3.f};
        /// Binning for the efficiency vs. phi plot
        plot_helpers::binning m_phi_binning{
            "Track Finding Efficiency vs. #phi;#phi", 40, -M_PI, M_PI};
        /// Binning for the efficiency vs. pT plot
        plot_helpers::binning m_pt_binning{
            "Track Finding Efficiency vs. p_{T};p_{T}", 40, 0.f, 120.f};

    };  // struct config

    /// Constructor with a configuration
    track_finding_analysis(const config& cfg);
    /// Destructor
    ~track_finding_analysis();

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
    /// Function checking whether two measurement objects match
    ///
    /// @param reco The reconstructed measurement
    /// @param truth The truth measurement
    ///
    /// @return Wehther the measurements match
    ///
    bool match(const measurement& reco, const measurement& truth) const;

    /// Configuration for the track finding efficiency measurement
    config m_config;
    /// Internal data used by the class
    std::unique_ptr<details::track_finding_analysis_data> m_data;

};  // class track_finding_analysis

}  // namespace traccc::performance
