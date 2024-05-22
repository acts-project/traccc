/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/performance/truth_filtering.hpp"

// System include(s).
#include <algorithm>
#include <cmath>

namespace traccc::performance {

truth_filtering::truth_filtering(const config& cfg) : m_config{cfg} {}

bool truth_filtering::passes(const particle& truth) const {

    // Calculate the truth particle's properties.
    const float truth_pt = getter::perp(truth.momentum);
    const float truth_eta = getter::eta(truth.momentum);

    // Check if the truth particle passes the filtering.
    return ((std::find(m_config.m_truth_pdgids.begin(),
                       m_config.m_truth_pdgids.end(),
                       truth.particle_type) != m_config.m_truth_pdgids.end()) &&
            (std::abs(truth_eta) <= m_config.m_truth_eta_max) &&
            (truth_pt >= m_config.m_truth_pt_min));
}

}  // namespace traccc::performance
