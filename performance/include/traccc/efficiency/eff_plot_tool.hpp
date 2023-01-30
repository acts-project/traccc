/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/particle.hpp"
#include "traccc/utils/helpers.hpp"
#include "traccc/utils/unit_vectors.hpp"

namespace traccc {

// Tools to make efficiency plots to show tracking efficiency.
// For the moment, the efficiency is taken as the fraction of successfully
// smoothed track over all tracks
class eff_plot_tool {
    public:
    /// @brief The nested configuration struct
    struct config {
        std::map<std::string, plot_helpers::binning> var_binning = {
            {"Eta", plot_helpers::binning("#eta", 40, -4, 4)},
            {"Phi", plot_helpers::binning("#phi", 100, -3.15, 3.15)},
            {"Pt", plot_helpers::binning("pT [GeV/c]", 40, 0, 100)}};
    };

    /// @brief Nested Cache struct
    struct eff_plot_cache {
        TEfficiency* track_eff_vs_pT{nullptr};   ///< Tracking efficiency vs pT
        TEfficiency* track_eff_vs_eta{nullptr};  ///< Tracking efficiency vs eta
        TEfficiency* track_eff_vs_phi{nullptr};  ///< Tracking efficiency vs phi
    };

    /// Constructor
    ///
    /// @param cfg Configuration struct
    /// @param lvl Message level declaration
    eff_plot_tool(const config& cfg) : m_cfg(cfg) {}

    /// @brief book the efficiency plots
    ///
    /// @param effPlotCache the cache for efficiency plots
    void book(std::string name, eff_plot_cache& cache) const {

        plot_helpers::binning b_phi = m_cfg.var_binning.at("Phi");
        plot_helpers::binning b_eta = m_cfg.var_binning.at("Eta");
        plot_helpers::binning b_pt = m_cfg.var_binning.at("Pt");

        // efficiency vs pT
        cache.track_eff_vs_pT = plot_helpers::book_eff(
            TString(name) + "_trackeff_vs_pT",
            "Tracking efficiency;Truth pT [GeV/c];Efficiency", b_pt);
        // efficiency vs eta
        cache.track_eff_vs_eta = plot_helpers::book_eff(
            TString(name) + "_trackeff_vs_eta",
            "Tracking efficiency;Truth #eta;Efficiency", b_eta);
        // efficiency vs phi
        cache.track_eff_vs_phi = plot_helpers::book_eff(
            TString(name) + "_trackeff_vs_phi",
            "Tracking efficiency;Truth #phi;Efficiency", b_phi);
    }

    /// @brief fill efficiency plots
    ///
    /// @param effPlotCache cache object for efficiency plots
    /// @param truthParticle the truth Particle
    /// @param status the reconstruction status
    void fill(eff_plot_cache& cache, const particle& truth_particle,
              bool status) const {

        const auto t_phi = phi(truth_particle.mom);
        const auto t_eta = eta(truth_particle.mom);
        const auto t_pT =
            getter::perp(vector2{truth_particle.mom[0], truth_particle.mom[1]});

        plot_helpers::fill_eff(cache.track_eff_vs_pT, t_pT, status);
        plot_helpers::fill_eff(cache.track_eff_vs_eta, t_eta, status);
        plot_helpers::fill_eff(cache.track_eff_vs_phi, t_phi, status);
    }

    /// @brief write the efficiency plots to file
    ///
    /// @param effPlotCache cache object for efficiency plots
    void write(const eff_plot_cache& cache) const {
        cache.track_eff_vs_pT->Write();
        cache.track_eff_vs_eta->Write();
        cache.track_eff_vs_phi->Write();
    }

    /// @brief delete the efficiency plots
    ///
    /// @param effPlotCache cache object for efficiency plots
    void clear(eff_plot_cache& cache) const {
        delete cache.track_eff_vs_pT;
        delete cache.track_eff_vs_eta;
        delete cache.track_eff_vs_phi;
    }

    private:
    config m_cfg;  ///< The Config class
};

}  // namespace traccc