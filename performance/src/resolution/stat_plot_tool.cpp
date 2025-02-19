/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "stat_plot_tool.hpp"

// ROOT include(s).
#ifdef TRACCC_HAVE_ROOT
#include <Math/ProbFuncMathCore.h>
#endif  // TRACCC_HAVE_ROOT

namespace traccc {

stat_plot_tool::stat_plot_tool(const stat_plot_tool_config& cfg) : m_cfg(cfg) {}

void stat_plot_tool::book(stat_plot_cache& cache) const {

    // Avoid unused variable warnings when building the code without ROOT.
    (void)cache;

#ifdef TRACCC_HAVE_ROOT
    plot_helpers::binning b_ndf = m_cfg.var_binning.at("ndf");
    plot_helpers::binning b_chi2 = m_cfg.var_binning.at("chi2");
    plot_helpers::binning b_reduced_chi2 = m_cfg.var_binning.at("reduced_chi2");
    plot_helpers::binning b_pval = m_cfg.var_binning.at("pval");
    plot_helpers::binning b_chi2_local = m_cfg.var_binning.at("chi2_local");
    cache.ndf_hist = plot_helpers::book_histo("ndf", "NDF", b_ndf);
    cache.chi2_hist = plot_helpers::book_histo("chi2", "Chi2", b_chi2);
    cache.reduced_chi2_hist =
        plot_helpers::book_histo("reduced_chi2", "Chi2/NDF", b_reduced_chi2);
    cache.pval_hist = plot_helpers::book_histo("pval", "p value", b_pval);
    for (unsigned int D = 1u; D <= 2u; D++) {
        cache.chi2_filtered_hist[D] = plot_helpers::book_histo(
            Form("chi2_%dD_filtered", D),
            Form("chi2 of %dD filtered parameters", D), b_chi2_local);
        cache.chi2_smoothed_hist[D] = plot_helpers::book_histo(
            Form("chi2_%dD_smoothed", D),
            Form("chi2 of %dD smoothed parameters", D), b_chi2_local);
        cache.pval_filtered_hist[D] = plot_helpers::book_histo(
            Form("pval_%dD_filtered", D),
            Form("p value of %dD filtered parameters", D), b_pval);
        cache.pval_smoothed_hist[D] = plot_helpers::book_histo(
            Form("pval_%dD_smoothed", D),
            Form("p value of %dD smoothed parameters", D), b_pval);
    }
#endif  // TRACCC_HAVE_ROOT
}

void stat_plot_tool::fill(stat_plot_cache& cache,
                          const finding_result& find_res) const {

    // Avoid unused variable warnings when building the code without ROOT.
    (void)cache;
    (void)find_res;

#ifdef TRACCC_HAVE_ROOT
    const auto& ndf = find_res.trk_quality.ndf;
    const auto& chi2 = find_res.trk_quality.chi2;
    cache.ndf_hist->Fill(ndf);
    cache.chi2_hist->Fill(chi2);
    cache.reduced_chi2_hist->Fill(chi2 / ndf);
    cache.pval_hist->Fill(ROOT::Math::chisquared_cdf_c(chi2, ndf));
#endif  // TRACCC_HAVE_ROOT
}

void stat_plot_tool::fill(
    stat_plot_cache& cache,
    const fitting_result<traccc::default_algebra>& fit_res) const {

    // Avoid unused variable warnings when building the code without ROOT.
    (void)cache;
    (void)fit_res;

#ifdef TRACCC_HAVE_ROOT
    const auto& ndf = fit_res.trk_quality.ndf;
    const auto& chi2 = fit_res.trk_quality.chi2;
    cache.ndf_hist->Fill(ndf);
    cache.chi2_hist->Fill(chi2);
    cache.reduced_chi2_hist->Fill(chi2 / ndf);
    cache.pval_hist->Fill(ROOT::Math::chisquared_cdf_c(chi2, ndf));
#endif  // TRACCC_HAVE_ROOT
}

void stat_plot_tool::fill(
    stat_plot_cache& cache,
    const track_state<traccc::default_algebra>& trk_state) const {

    // Avoid unused variable warnings when building the code without ROOT.
    (void)cache;
    (void)trk_state;

#ifdef TRACCC_HAVE_ROOT
    const unsigned int D = trk_state.get_measurement().meas_dim;
    const auto filtered_chi2 = trk_state.filtered_chi2();
    const auto smoothed_chi2 = trk_state.smoothed_chi2();
    cache.chi2_filtered_hist[D]->Fill(filtered_chi2);
    cache.chi2_smoothed_hist[D]->Fill(smoothed_chi2);
    cache.pval_filtered_hist[D]->Fill(
        ROOT::Math::chisquared_cdf_c(filtered_chi2, D));
    cache.pval_smoothed_hist[D]->Fill(
        ROOT::Math::chisquared_cdf_c(smoothed_chi2, D));
#endif  // TRACCC_HAVE_ROOT
}

void stat_plot_tool::write(const stat_plot_cache& cache) const {

    // Avoid unused variable warnings when building the code without ROOT.
    (void)cache;

#ifdef TRACCC_HAVE_ROOT
    cache.ndf_hist->Write();
    cache.chi2_hist->Write();
    cache.reduced_chi2_hist->Write();
    cache.pval_hist->Write();
    cache.ndf_hist->Write();
    cache.chi2_hist->Write();
    cache.reduced_chi2_hist->Write();
    cache.pval_hist->Write();

    for (const auto& flt : cache.chi2_filtered_hist) {
        flt.second->Write();
    }
    for (const auto& smt : cache.chi2_smoothed_hist) {
        smt.second->Write();
    }
    for (const auto& flt : cache.pval_filtered_hist) {
        flt.second->Write();
    }
    for (const auto& smt : cache.pval_smoothed_hist) {
        smt.second->Write();
    }

#endif  // TRACCC_HAVE_ROOT
}

}  // namespace traccc
