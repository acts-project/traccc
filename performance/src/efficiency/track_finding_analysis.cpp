/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/efficiency/track_finding_analysis.hpp"

#include "traccc/performance/details/is_same_object.hpp"

// System include(s).
#include <cmath>
#include <iostream>

namespace traccc::performance {

track_finding_analysis::track_finding_analysis() {}

void track_finding_analysis::analyze(
    const track_candidate_container_types::const_view& reco_particles_view,
    const particle_container_types::const_view& truth_particles_view) {

    // Construct device containers around the views.
    track_candidate_container_types::const_device reco_particles{
        reco_particles_view};
    particle_container_types::const_device truth_particles{
        truth_particles_view};

    // Helper type.
    using size_type = track_candidate_container_types::const_device::size_type;

    // Print the properties of the reco particles.
    for (size_type i_reco = 0; i_reco < reco_particles.size(); ++i_reco) {

        // The reconstructed particle.
        track_candidate_container_types::const_device::const_element_view
            reco_particle = reco_particles.at(i_reco);

        std::cout << "Reco particle eta: "
                  << getter::eta(reco_particle.header.mom())
                  << " phi: " << reco_particle.header.phi()
                  << " pT: " << reco_particle.header.pT() << std::endl;
        for (const measurement& meas : reco_particle.items) {
            std::cout << "   Measurement local[0]:" << meas.local[0]
                      << " local[1]: " << meas.local[1]
                      << " variance[0]: " << meas.variance[0]
                      << " variance[1]: " << meas.variance[1]
                      << " meas_dim: " << meas.meas_dim
                      << " surface link: " << meas.surface_link << std::endl;
        }
    }

    // Loop over the truth particles.
    for (size_type i_truth = 0; i_truth < truth_particles.size(); ++i_truth) {

        // The truth particle.
        particle_container_types::const_device::const_element_view
            truth_particle = truth_particles.at(i_truth);

        // Require it to be a muon.
        if (std::abs(truth_particle.header.particle_type) != 13) {
            continue;
        }

        // Look for a matching reconstructed particle.
        bool found = false;
        for (size_type i_reco = 0; i_reco < reco_particles.size(); ++i_reco) {

            // The reconstructed particle.
            track_candidate_container_types::const_device::const_element_view
                reco_particle = reco_particles.at(i_reco);

            // Check if it matches the truth particle.
            if (match(reco_particle, truth_particle)) {
                found = true;
                break;
            }
        }

        std::cout << "Truth particle eta: "
                  << getter::eta(truth_particle.header.momentum)
                  << " phi: " << getter::phi(truth_particle.header.momentum)
                  << " pT: " << getter::perp(truth_particle.header.momentum)
                  << " found: " << found << std::endl;
        for (const measurement& meas : truth_particle.items) {
            std::cout << "   Measurement local[0]:" << meas.local[0]
                      << " local[1]: " << meas.local[1]
                      << " variance[0]: " << meas.variance[0]
                      << " variance[1]: " << meas.variance[1]
                      << " meas_dim: " << meas.meas_dim
                      << " surface link: " << meas.surface_link << std::endl;
        }
    }
}

bool track_finding_analysis::match(
    const track_candidate_container_types::const_device::const_element_view&
        reco,
    const particle_container_types::const_device::const_element_view& truth)
    const {

    // Check how many true and reconstructed measurements match from the two
    // particles.
    unsigned int n_matches = 0;
    for (const measurement& reco_meas : reco.items) {
        for (const measurement& truth_meas : truth.items) {
            if (details::is_same_object(reco_meas, 0.1)(truth_meas)) {
                ++n_matches;
            }
        }
    }

    // Calculate a match rate.
    float match_rate =
        static_cast<float>(n_matches) / static_cast<float>(reco.items.size());
    std::cout << "Match rate: " << match_rate << std::endl;

    // Define the match relatively naively.
    return (match_rate > 0.8f);
}

}  // namespace traccc::performance
