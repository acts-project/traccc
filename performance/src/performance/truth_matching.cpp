/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/performance/truth_matching.hpp"

// System include(s).
#include <cmath>

namespace traccc::performance {

truth_matching::truth_matching(const config& cfg) : m_config{cfg} {}

bool truth_matching::matches(
    const seed& reco,
    const spacepoint_collection_types::const_view spacepoints_view,
    const particle_container_types::const_device::const_element_view& truth)
    const {

    // Create a device collection for the spacepoints.
    const spacepoint_collection_types::const_device spacepoints{
        spacepoints_view};

    // The number of matching measurements.
    unsigned int n_matches = 0;

    // Loop over the measurements of the truth particle.
    for (const measurement& truth_meas : truth.items) {
        // Check one-by-one if the measurements that the seed is made of,
        // match the truth measurements.
        const measurement& bottom_meas = spacepoints.at(reco.spB_link).meas;
        if (matches(bottom_meas, truth_meas)) {
            ++n_matches;
        }
        const measurement& middle_meas = spacepoints.at(reco.spM_link).meas;
        if (matches(middle_meas, truth_meas)) {
            ++n_matches;
        }
        const measurement& top_meas = spacepoints.at(reco.spT_link).meas;
        if (matches(top_meas, truth_meas)) {
            ++n_matches;
        }
    }

    // Calculate a match rate, normalized to the number of seed measurements.
    const float match_rate = static_cast<float>(n_matches) / 3.f;

    // Define the match relatively naively.
    return (match_rate >= m_config.m_measurement_match);
}

bool truth_matching::matches(
    const track_candidate_container_types::const_device::const_element_view&
        reco,
    const particle_container_types::const_device::const_element_view& truth)
    const {

    // Check how many true and reconstructed measurements match from the two
    // particles.
    unsigned int n_matches = 0;
    for (const measurement& reco_meas : reco.items) {
        for (const measurement& truth_meas : truth.items) {
            if (matches(reco_meas, truth_meas)) {
                ++n_matches;
            }
        }
    }

    // Calculate a match rate, normalized to the number of reconstructed
    // measurements.
    const float match_rate =
        static_cast<float>(n_matches) / static_cast<float>(reco.items.size());

    // Define the match relatively naively.
    return (match_rate >= m_config.m_measurement_match);
}

bool truth_matching::matches(const measurement& reco,
                             const measurement& truth) const {

    return ((reco.module_link == truth.module_link) &&
            (std::abs(reco.local[0] - truth.local[0]) <
             m_config.m_measurement_uncertainty) &&
            (std::abs(reco.local[1] - truth.local[1]) <
             m_config.m_measurement_uncertainty) &&
            (std::abs(reco.variance[0] - truth.variance[0]) <
             m_config.m_measurement_uncertainty) &&
            (std::abs(reco.variance[1] - truth.variance[1]) <
             m_config.m_measurement_uncertainty));
}

}  // namespace traccc::performance
