/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/efficiency/track_finding_efficiency.hpp"

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

struct track_finding_efficiency_data {
#ifdef TRACCC_HAVE_ROOT
    std::unique_ptr<TEfficiency> m_eta;
    std::unique_ptr<TEfficiency> m_phi;
    std::unique_ptr<TEfficiency> m_pt;
#endif  // TRACCC_HAVE_ROOT
};  // struct track_finding_efficiency_data

}  // namespace details

track_finding_efficiency::track_finding_efficiency(
    const config& eff_cfg, const truth_filtering::config& filter_cfg,
    const truth_matching::config& matching_cfg)
    : truth_filtering{filter_cfg},
      truth_matching{matching_cfg},
      m_config{eff_cfg},
      m_data{std::make_unique<details::track_finding_efficiency_data>()} {

#ifdef TRACCC_HAVE_ROOT
    // Helper lambda for creating the TEfficiency objects.
    auto make_teff = [](std::string_view name,
                        const plot_helpers::binning& bins) {
        return std::make_unique<TEfficiency>(name.data(), bins.title.c_str(),
                                             bins.n_bins, bins.min, bins.max);
    };

    m_data->m_eta = make_teff("track_finding_eff_eta", m_config.m_eta_binning);
    m_data->m_phi = make_teff("track_finding_eff_phi", m_config.m_phi_binning);
    m_data->m_pt = make_teff("track_finding_eff_pt", m_config.m_pt_binning);
#endif  // TRACCC_HAVE_ROOT
}

track_finding_efficiency::~track_finding_efficiency() {

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

void track_finding_efficiency::analyze(
    [[maybe_unused]] const track_candidate_container_types::const_view&
        reco_particles_view,
    [[maybe_unused]] const particle_container_types::const_view&
        truth_particles_view) {

#ifdef TRACCC_HAVE_ROOT
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

        // Check if the truth particle should be considered.
        if (passes(truth_particle.header) == false) {
            continue;
        }

        // Look for a matching reconstructed particle.
        [[maybe_unused]] bool found = false;
        for (size_type i_reco = 0; i_reco < reco_particles.size(); ++i_reco) {

            // The reconstructed particle.
            track_candidate_container_types::const_device::const_element_view
                reco_particle = reco_particles.at(i_reco);

            // Check if it matches the truth particle.
            if (matches(reco_particle, truth_particle)) {
                found = true;
                break;
            }
        }

        // Fill the efficiency objects.
        m_data->m_eta->Fill(found, getter::eta(truth_particle.header.momentum));
        m_data->m_phi->Fill(found, getter::phi(truth_particle.header.momentum));
        m_data->m_pt->Fill(found, getter::perp(truth_particle.header.momentum));
    }
#endif  // TRACCC_HAVE_ROOT
}

}  // namespace traccc::performance
