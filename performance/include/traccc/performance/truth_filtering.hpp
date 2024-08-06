/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/edm/particle.hpp"

// System include(s).
#include <vector>

namespace traccc::performance {

/// Class providing truth particle filtering
class truth_filtering {

    public:
    /// Configuration for the truth particle filtering.
    struct config {
        /// Truth particle PDG IDs to take into consideration
        std::vector<int> m_truth_pdgids{-13, 13};
        /// Minimum truth pT to take into consideration
        float m_truth_pt_min{500.f * unit<float>::MeV};
        /// Maximum (absolute) truth pseudorapidity to take into consideration
        float m_truth_eta_max{3.0f};
    };  // struct config

    /// Constructor with a configuration
    truth_filtering(const config& cfg);

    /// Function checking whether a truth particle passes the filter
    ///
    /// @param truth The truth particle
    ///
    /// @return Whether the truth particle passes the filter
    ///
    bool passes(const particle& truth) const;

    private:
    /// Configuration for the truth particle filtering.
    config m_config;

};  // class truth_filtering

}  // namespace traccc::performance
