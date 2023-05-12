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
    cache.ndf_hist = plot_helpers::book_histo("ndf", "NDF", b_ndf);
    cache.chi2_hist = plot_helpers::book_histo("chi2", "Chi2", b_chi2);
    cache.reduced_chi2_hist =
        plot_helpers::book_histo("reduced_chi2", "Chi2/NDF", b_reduced_chi2);
    cache.pval_hist = plot_helpers::book_histo("pval", "p value", b_pval);
#endif  // TRACCC_HAVE_ROOT
}

void stat_plot_tool::fill(stat_plot_cache& cache,
                          const fitter_info<transform3>& fit_info) const {

    // Avoid unused variable warnings when building the code without ROOT.
    (void)cache;

#ifdef TRACCC_HAVE_ROOT
    const auto& ndf = fit_info.ndf;
    const auto& chi2 = fit_info.chi2;
    cache.ndf_hist->Fill(ndf);
    cache.chi2_hist->Fill(chi2);
    cache.reduced_chi2_hist->Fill(chi2 / ndf);
    cache.pval_hist->Fill(ROOT::Math::chisquared_cdf_c(chi2, ndf));
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
#endif  // TRACCC_HAVE_ROOT
}

}  // namespace traccc
