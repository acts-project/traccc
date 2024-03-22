/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "../utils/helpers.hpp"

// Project include(s).
#include "traccc/edm/particle.hpp"
#include "traccc/edm/track_state.hpp"

// ROOT include(s).
#ifdef TRACCC_HAVE_ROOT
#include <TFile.h>
#include <TTree.h>
#endif  // TRACCC_HAVE_ROOT

// System include(s).
#include <string_view>

namespace traccc {

class event_tree_tool {

    public:
    /// @brief Nested Cache struct
    struct event_tree_cache {
#ifdef TRACCC_HAVE_ROOT
        // Particles truth information
        std::unique_ptr<TTree> ptc_tree =
            std::make_unique<TTree>("ptc_tree", "Tree for truth particles");

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
    };

    void setup(event_tree_cache& cache) {
        (void)cache;

#ifdef TRACCC_HAVE_ROOT
        cache.ptc_tree->Branch("vx", &cache.vx, "vx/F");
        cache.ptc_tree->Branch("vy", &cache.vy, "vy/F");
        cache.ptc_tree->Branch("vz", &cache.vz, "vz/F");
        cache.ptc_tree->Branch("vt", &cache.vt, "vt/F");
        cache.ptc_tree->Branch("phi", &cache.phi, "phi/F");
        cache.ptc_tree->Branch("theta", &cache.theta, "theta/F");
        cache.ptc_tree->Branch("charge", &cache.charge, "charge/F");
        cache.ptc_tree->Branch("qop", &cache.qop, "qop/F");
        cache.ptc_tree->Branch("qopT", &cache.qopT, "qopT/F");
        cache.ptc_tree->Branch("qopz", &cache.qopz, "qopz/F");

        cache.fit_tree->Branch("fit_loc0", &cache.fit_loc0, "fit_loc0/F");
        cache.fit_tree->Branch("fit_loc1", &cache.fit_loc1, "fit_loc1/F");
        cache.fit_tree->Branch("fit_phi", &cache.fit_phi, "fit_phi/F");
        cache.fit_tree->Branch("fit_theta", &cache.fit_theta, "fit_theta/F");
        cache.fit_tree->Branch("fit_time", &cache.fit_time, "fit_time/F");
        cache.fit_tree->Branch("fit_qop", &cache.fit_qop, "fit_qop/F");
        cache.fit_tree->Branch("fit_qopT", &cache.fit_qopT, "fit_qopT/F");
        cache.fit_tree->Branch("fit_qopz", &cache.fit_qopz, "fit_qopz/F");
        cache.fit_tree->Branch("fit_ndf", &cache.fit_ndf, "fit_ndf/F");
        cache.fit_tree->Branch("fit_chi2", &cache.fit_chi2, "fit_chi2/F");
#endif  // TRACCC_HAVE_ROOT
    }

    void fill(event_tree_cache& cache, const particle& truth_particle) const {

        cache.vx = truth_particle.pos[0];
        cache.vy = truth_particle.pos[1];
        cache.vz = truth_particle.pos[2];
        cache.vt = truth_particle.time;
        cache.phi = getter::phi(truth_particle.mom);
        cache.theta = getter::theta(truth_particle.mom);
        cache.qop = truth_particle.charge / getter::norm(truth_particle.mom);
        cache.qopT = truth_particle.charge / getter::perp(truth_particle.mom);
        cache.qopz = truth_particle.charge / std::abs(truth_particle.mom[2]);
        cache.charge = truth_particle.charge;

#ifdef TRACCC_HAVE_ROOT
        cache.ptc_tree->Fill();
#endif  // TRACCC_HAVE_ROOT
    }

    void fill(
        event_tree_cache& cache,
        const traccc::track_state_container_types::const_device::element_view&
            fit_track) const {
        const auto& vec = fit_track.header.fit_params.vector();

        cache.fit_loc0 = getter::element(vec, e_bound_loc0, 0u);
        cache.fit_loc1 = getter::element(vec, e_bound_loc1, 0u);
        cache.fit_phi = getter::element(vec, e_bound_phi, 0u);
        cache.fit_theta = getter::element(vec, e_bound_theta, 0u);
        cache.fit_time = getter::element(vec, e_bound_time, 0u);
        cache.fit_qop = getter::element(vec, e_bound_qoverp, 0u);
        cache.fit_qopT = fit_track.header.fit_params.qopT();
        cache.fit_qopz = fit_track.header.fit_params.qopz();
        cache.fit_ndf = fit_track.header.ndf;
        cache.fit_chi2 = fit_track.header.chi2;

#ifdef TRACCC_HAVE_ROOT
        cache.fit_tree->Fill();
#endif  // TRACCC_HAVE_ROOT
    }

    void write(const event_tree_cache& cache) const {

        // Avoid unused variable warnings when building the code without ROOT.
        (void)cache;

#ifdef TRACCC_HAVE_ROOT
        cache.ptc_tree->Write();
        cache.fit_tree->Write();
#endif  // TRACCC_HAVE_ROOT
    }
};

}  // namespace traccc