/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/particle.hpp"
#include "traccc/efficiency/helpers.hpp"
#include "traccc/utils/unit_vectors.hpp"

namespace traccc {

// Tools to make duplication rate and duplication number plots to show tracking
// duplication.
//
// The duplication is investigated for those truth-matched reco tracks. If there
// are a few reco tracks matched to the same truth particle, the reco track with
// the highest matching probability is tagges as 'real' and the others are
// 'duplicated'.
class duplication_plot_tool {
    public:
    /// @brief The nested configuration struct
    struct config {
        std::map<std::string, plot_helpers::binning> var_binning = {
            {"Eta", plot_helpers::binning("#eta", 40, -4, 4)},
            {"Phi", plot_helpers::binning("#phi", 100, -3.15, 3.15)},
            {"Pt", plot_helpers::binning("pT [GeV/c]", 40, 0, 100)},
            {"Num", plot_helpers::binning("N", 30, -0.5, 29.5)}};
    };

    /// @brief Nested Cache struct
    struct duplication_plot_cache {
        TProfile* n_duplicated_vs_pT;   ///< Number of duplicated tracks vs pT
        TProfile* n_duplicated_vs_eta;  ///< Number of duplicated tracks vs eta
        TProfile* n_duplicated_vs_phi;  ///< Number of duplicated tracks vs phi
        TEfficiency*
            duplication_rate_vs_pT;  ///< Tracking duplication rate vs pT
        TEfficiency*
            duplication_rate_vs_eta;  ///< Tracking duplication rate vs eta
        TEfficiency*
            duplication_rate_vs_phi;  ///< Tracking duplication rate vs phi
    };

    /// Constructor
    ///
    /// @param cfg Configuration struct
    duplication_plot_tool(const config& cfg) : m_cfg(cfg) {}

    /// @brief book the duplication plots
    ///
    /// @param duplicationPlotCache the cache for duplication plots
    void book(std::string name, duplication_plot_cache& cache) const {
        plot_helpers::binning b_pt = m_cfg.var_binning.at("Pt");
        plot_helpers::binning b_eta = m_cfg.var_binning.at("Eta");
        plot_helpers::binning b_phi = m_cfg.var_binning.at("Phi");
        plot_helpers::binning b_num = m_cfg.var_binning.at("Num");

        // duplication rate vs pT
        cache.duplication_rate_vs_pT = plot_helpers::book_eff(
            TString(name) + "_duplicationRate_vs_pT",
            "Duplication rate;pT [GeV/c];Duplication rate", b_pt);
        // duplication rate vs eta
        cache.duplication_rate_vs_eta = plot_helpers::book_eff(
            TString(name) + "_duplicationRate_vs_eta",
            "Duplication rate;#eta;Duplication rate", b_eta);
        // duplication rate vs phi
        cache.duplication_rate_vs_phi = plot_helpers::book_eff(
            TString(name) + "_duplicationRate_vs_phi",
            "Duplication rate;#phi;Duplication rate", b_phi);

        // duplication number vs pT
        cache.n_duplicated_vs_pT = plot_helpers::book_prof(
            TString(name) + "_nDuplicated_vs_pT",
            "Number of duplicated track candidates", b_pt, b_num);
        // duplication number vs eta
        cache.n_duplicated_vs_eta = plot_helpers::book_prof(
            TString(name) + "_nDuplicated_vs_eta",
            "Number of duplicated track candidates", b_eta, b_num);
        // duplication number vs phi
        cache.n_duplicated_vs_phi = plot_helpers::book_prof(
            TString(name) + "_nDuplicated_vs_phi",
            "Number of duplicated track candidates", b_phi, b_num);
    }

    /// @brief fill number of duplicated tracks for a truth particle seed
    ///
    /// @param duplicationPlotCache cache object for duplication plots
    /// @param truthParticle the truth Particle
    /// @param nDuplicatedTracks the number of duplicated tracks
    void fill(duplication_plot_cache& cache, const particle& truth_particle,
              size_t n_duplicated_tracks) const {
        const auto t_phi = phi(truth_particle.mom);
        const auto t_eta = eta(truth_particle.mom);
        const auto t_pT =
            getter::perp(vector2{truth_particle.mom[0], truth_particle.mom[1]});

        plot_helpers::fill_prof(cache.n_duplicated_vs_pT, t_pT,
                                n_duplicated_tracks);
        plot_helpers::fill_prof(cache.n_duplicated_vs_eta, t_eta,
                                n_duplicated_tracks);
        plot_helpers::fill_prof(cache.n_duplicated_vs_phi, t_phi,
                                n_duplicated_tracks);
    }

    /// @brief write the duplication plots to file
    ///
    /// @param duplicationPlotCache cache object for duplication plots
    void write(const duplication_plot_cache& cache) const {
        cache.duplication_rate_vs_pT->Write();
        cache.duplication_rate_vs_eta->Write();
        cache.duplication_rate_vs_phi->Write();
        cache.n_duplicated_vs_pT->Write();
        cache.n_duplicated_vs_eta->Write();
        cache.n_duplicated_vs_phi->Write();
    }

    /// @brief delete the duplication plots
    ///
    /// @param duplicationPlotCache cache object for duplication plots
    void clear(duplication_plot_cache& cache) const {
        delete cache.duplication_rate_vs_pT;
        delete cache.duplication_rate_vs_eta;
        delete cache.duplication_rate_vs_phi;
        delete cache.n_duplicated_vs_pT;
        delete cache.n_duplicated_vs_eta;
        delete cache.n_duplicated_vs_phi;
    }

    private:
    config m_cfg;  ///< The Config class
};

}  // namespace traccc