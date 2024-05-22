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
#include "traccc/edm/seed.hpp"
#include "traccc/edm/spacepoint.hpp"

// System include(s).
#include <cmath>
#include <memory>
#include <string>

namespace traccc::performance {

// Forward declaration(s).
namespace details {
struct seed_finding_efficiency_data;
}

/// Tool for measuring the seed finding efficiency
class seed_finding_efficiency : public truth_filtering, public truth_matching {

    public:
    /// Configuration for the efficiency analysis
    struct config {

        /// Name of the output file to write
        std::string m_output_name{"seed_finding_efficiency.root"};

        /// Binning for the efficiency vs. eta plot
        plot_helpers::binning m_eta_binning{
            "Seed Finding Efficiency vs. #eta;#eta", 40, -3.f, 3.f};
        /// Binning for the efficiency vs. phi plot
        plot_helpers::binning m_phi_binning{
            "Seed Finding Efficiency vs. #phi;#phi", 40, -M_PI, M_PI};
        /// Binning for the efficiency vs. pT plot
        plot_helpers::binning m_pt_binning{
            "Seed Finding Efficiency vs. p_{T};p_{T}", 40, 0.f, 120.f};

    };  // struct config

    /// Constructor with a configuration
    seed_finding_efficiency(const config& eff_cfg,
                            const truth_filtering::config& filter_cfg,
                            const truth_matching::config& matching_cfg);
    /// Destructor
    ~seed_finding_efficiency();

    /// Analyze the tracks found in a specific event.
    ///
    /// @param seeds The reconstructed particle seeds
    /// @param spacepoints The spacepoints that the seeds are made of
    /// @param truth The truth particles
    ///
    void analyze(const seed_collection_types::const_view& seeds,
                 const spacepoint_collection_types::const_view& spacepoints,
                 const particle_container_types::const_view& truth);

    private:
    /// Configuration for the seed finding efficiency measurement
    config m_config;
    /// Internal data used by the class
    std::unique_ptr<details::seed_finding_efficiency_data> m_data;

};  // class seed_finding_efficiency

}  // namespace traccc::performance
