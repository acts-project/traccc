/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/performance/truth_filtering.hpp"
#include "traccc/performance/truth_matching.hpp"
#include "traccc/utils/helpers.hpp"

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/edm/particle.hpp"
#include "traccc/edm/track_candidate.hpp"

// System include(s).
#include <cmath>
#include <memory>
#include <string>

namespace traccc::performance {

// Forward declaration(s).
namespace details {
struct track_finding_efficiency_data;
}

/// Tool for analyzing the track finding efficiency
class track_finding_efficiency : public truth_filtering, public truth_matching {

    public:
    /// Configuration for the efficiency analysis
    struct config {

        /// Name of the output file to write
        std::string m_output_name{"track_finding_efficiency.root"};

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
    track_finding_efficiency(const config& eff_cfg,
                             const truth_filtering::config& filter_cfg,
                             const truth_matching::config& matching_cfg);
    /// Destructor
    ~track_finding_efficiency();

    /// Analyze the tracks found in a specific event.
    ///
    /// @param reco The reconstructed particle tracks
    /// @param truth The truth particles
    ///
    void analyze(const track_candidate_container_types::const_view& reco,
                 const particle_container_types::const_view& truth);

    private:
    /// Configuration for the track finding efficiency measurement
    config m_config;
    /// Internal data used by the class
    std::unique_ptr<details::track_finding_efficiency_data> m_data;

};  // class track_finding_efficiency

}  // namespace traccc::performance
