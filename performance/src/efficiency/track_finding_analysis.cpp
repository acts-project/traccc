/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/efficiency/track_finding_analysis.hpp"

#ifdef TRACCC_HAVE_ROOT
// ROOT include(s).
#include <TEfficiency.h>
#include <TFile.h>
#endif  // TRACCC_HAVE_ROOT

// System include(s).
#include <algorithm>
#include <cmath>
#include <iostream>

namespace traccc::performance {

namespace details {

struct track_finding_analysis_data {
#ifdef TRACCC_HAVE_ROOT
    std::unique_ptr<TEfficiency> m_eta;
    std::unique_ptr<TEfficiency> m_phi;
    std::unique_ptr<TEfficiency> m_pt;
#endif  // TRACCC_HAVE_ROOT
};  // struct track_finding_analysis_data

}  // namespace details

track_finding_analysis::track_finding_analysis(const config& cfg)
    : m_config{cfg},
      m_data{std::make_unique<details::track_finding_analysis_data>()} {

#ifdef TRACCC_HAVE_ROOT
    // Helper lambda for creating the TEfficiency objects.
    auto make_teff = [](std::string_view name,
                        const plot_helpers::binning& bins) {
        return std::make_unique<TEfficiency>(name.data(), bins.title.c_str(),
                                             bins.n_bins, bins.min, bins.max);
    };

    m_data->m_eta = make_teff("track_finding_eff_eta", cfg.m_eta_binning);
    m_data->m_phi = make_teff("track_finding_eff_phi", cfg.m_phi_binning);
    m_data->m_pt = make_teff("track_finding_eff_pt", cfg.m_pt_binning);
#endif  // TRACCC_HAVE_ROOT
}

track_finding_analysis::~track_finding_analysis() {

#ifdef TRACCC_HAVE_ROOT
    // Open the output file.
    std::unique_ptr<TFile> ofile{
        TFile::Open(m_config.m_output_name.c_str(), "RECREATE")};

    // Save all the efficiency objects into it.
    m_data->m_eta->Write();
    m_data->m_phi->Write();
    m_data->m_pt->Write();

    // Tell the user what happened.
    std::cout << "Saved track finding efficiency plots into: "
              << m_config.m_output_name << std::endl;
#endif  // TRACCC_HAVE_ROOT
}

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

    // Loop over the truth particles.
    for (size_type i_truth = 0; i_truth < truth_particles.size(); ++i_truth) {

        // The truth particle.
        particle_container_types::const_device::const_element_view
            truth_particle = truth_particles.at(i_truth);

        // Calculate the truth particle's properties.
        const scalar truth_eta = getter::eta(truth_particle.header.momentum);
        const scalar truth_phi = getter::phi(truth_particle.header.momentum);
        const scalar truth_pt = getter::perp(truth_particle.header.momentum);

        // Check if the truth particle should be considered.
        if ((std::find(m_config.m_truth_pdgid.begin(),
                       m_config.m_truth_pdgid.end(),
                       truth_particle.header.particle_type) ==
             m_config.m_truth_pdgid.end()) ||
            (std::abs(truth_eta) > m_config.m_truth_eta_max) ||
            (truth_pt < m_config.m_truth_pt_min)) {
            continue;
        }

        // Look for a matching reconstructed particle.
        [[maybe_unused]] bool found = false;
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

#ifdef TRACCC_HAVE_ROOT
        // Fill the efficiency objects.
        m_data->m_eta->Fill(found, truth_eta);
        m_data->m_phi->Fill(found, truth_phi);
        m_data->m_pt->Fill(found, truth_pt);
#endif  // TRACCC_HAVE_ROOT
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
            if (match(reco_meas, truth_meas)) {
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

bool track_finding_analysis::match(const measurement& reco,
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
