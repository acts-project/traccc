/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/event/track_property_writer.hpp"

// ROOT include(s).
#ifdef TRACCC_HAVE_ROOT
#include <TFile.h>
#include <TTree.h>
#endif  // TRACCC_HAVE_ROOT

namespace traccc {

namespace details {

struct track_property_writer_data {

    void setup() {
#ifdef TRACCC_HAVE_ROOT
        truth_tree->Branch("truth_vx", &vx, "vx/F");
        truth_tree->Branch("truth_vy", &vy, "vy/F");
        truth_tree->Branch("truth_vz", &vz, "vz/F");
        truth_tree->Branch("truth_vt", &vt, "vt/F");
        truth_tree->Branch("truth_phi", &phi, "phi/F");
        truth_tree->Branch("truth_theta", &theta, "theta/F");
        truth_tree->Branch("truth_charge", &charge, "charge/F");
        truth_tree->Branch("truth_qop", &qop, "qop/F");
        truth_tree->Branch("truth_qopT", &qopT, "qopT/F");
        truth_tree->Branch("truth_qopz", &qopz, "qopz/F");

        fit_tree->Branch("fit_loc0", &fit_loc0, "fit_loc0/F");
        fit_tree->Branch("fit_loc1", &fit_loc1, "fit_loc1/F");
        fit_tree->Branch("fit_phi", &fit_phi, "fit_phi/F");
        fit_tree->Branch("fit_theta", &fit_theta, "fit_theta/F");
        fit_tree->Branch("fit_time", &fit_time, "fit_time/F");
        fit_tree->Branch("fit_qop", &fit_qop, "fit_qop/F");
        fit_tree->Branch("fit_qopT", &fit_qopT, "fit_qopT/F");
        fit_tree->Branch("fit_qopz", &fit_qopz, "fit_qopz/F");
        fit_tree->Branch("fit_ndf", &fit_ndf, "fit_ndf/F");
        fit_tree->Branch("fit_chi2", &fit_chi2, "fit_chi2/F");
#endif  // TRACCC_HAVE_ROOT
    }

    void fill(const particle& truth_particle) {
        vx = truth_particle.pos[0];
        vy = truth_particle.pos[1];
        vz = truth_particle.pos[2];
        vt = truth_particle.time;
        phi = getter::phi(truth_particle.mom);
        theta = getter::theta(truth_particle.mom);
        qop = truth_particle.charge / getter::norm(truth_particle.mom);
        qopT = truth_particle.charge / getter::perp(truth_particle.mom);
        qopz = truth_particle.charge / std::abs(truth_particle.mom[2]);
        charge = truth_particle.charge;

#ifdef TRACCC_HAVE_ROOT
        truth_tree->Fill();
#endif  // TRACCC_HAVE_ROOT
    }

    void fill(
        const traccc::track_state_container_types::const_device::element_view&
            fit_track) {
        const auto& vec = fit_track.header.fit_params.vector();

        fit_ndf = fit_track.header.ndf;
        fit_chi2 = fit_track.header.chi2;
        if (fit_ndf > 0) {
            fit_loc0 = getter::element(vec, e_bound_loc0, 0u);
            fit_loc1 = getter::element(vec, e_bound_loc1, 0u);
            fit_phi = getter::element(vec, e_bound_phi, 0u);
            fit_theta = getter::element(vec, e_bound_theta, 0u);
            fit_time = getter::element(vec, e_bound_time, 0u);
            fit_qop = getter::element(vec, e_bound_qoverp, 0u);
            fit_qopT = fit_track.header.fit_params.qopT();
            fit_qopz = fit_track.header.fit_params.qopz();
        }
#ifdef TRACCC_HAVE_ROOT
        fit_tree->Fill();
#endif  // TRACCC_HAVE_ROOT
    }

    void write() const {
#ifdef TRACCC_HAVE_ROOT
        truth_tree->Write();
        fit_tree->Write();
#endif  // TRACCC_HAVE_ROOT
    }

#ifdef TRACCC_HAVE_ROOT
    // Particles truth information
    std::unique_ptr<TTree> truth_tree =
        std::make_unique<TTree>("truth_tree", "Tree for truth particles");

    // Fit results information
    std::unique_ptr<TTree> fit_tree =
        std::make_unique<TTree>("fit_tree", "Tree for fitting results");
#endif  // TRACCC_HAVE_ROOT

    float vx;
    float vy;
    float vz;
    float vt;
    float phi;
    float theta;
    float qop;
    float qopT;
    float qopz;
    float charge;

    float fit_loc0;
    float fit_loc1;
    float fit_phi;
    float fit_theta;
    float fit_time;
    float fit_qop;
    float fit_qopT;
    float fit_qopz;
    float fit_ndf;
    float fit_chi2;

};  // struct track_property_writer_data

}  // namespace details

track_property_writer::track_property_writer(const config& cfg)
    : m_cfg(cfg),
      m_data(std::make_unique<details::track_property_writer_data>()) {

    m_data->setup();
}

track_property_writer::~track_property_writer() {}

void track_property_writer::write(
    const track_state_container_types::const_view& track_states_view,
    const event_map2& evt_map) {

    for (auto const& [key, ptc] : evt_map.ptc_map) {
        m_data->fill(ptc);
    }

    // Iterate over the track state container.
    track_state_container_types::const_device track_states(track_states_view);

    const unsigned int n_tracks = track_states.size();

    for (unsigned int i = 0; i < n_tracks; i++) {
        m_data->fill(track_states.at(i));
    }
}

void track_property_writer::finalize() {

#ifdef TRACCC_HAVE_ROOT
    // Open the output file.
    std::unique_ptr<TFile> ofile(
        TFile::Open(m_cfg.file_path.c_str(), m_cfg.file_mode.c_str()));
    if ((!ofile) || ofile->IsZombie()) {
        throw std::runtime_error("Could not open output file \"" +
                                 m_cfg.file_path + "\" in mode \"" +
                                 m_cfg.file_mode + "\"");
    }
    ofile->cd();
#else
    std::cout << "ROOT file \"" << m_cfg.file_path << "\" is NOT created"
              << std::endl;
#endif  // TRACCC_HAVE_ROOT

    m_data->write();
}

}  // namespace traccc
