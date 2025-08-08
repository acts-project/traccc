/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/utils/prob.hpp"

namespace traccc {

template <typename T>
void stat_plot_tool::fill(stat_plot_cache& cache,
                          const edm::track_candidate<T>& find_res) const {

    // Avoid unused variable warnings when building the code without ROOT.
    (void)cache;
    (void)find_res;

#ifdef TRACCC_HAVE_ROOT
    const auto& ndf = find_res.ndf();
    const auto& chi2 = find_res.chi2();
    cache.ndf_hist->Fill(ndf);
    cache.chi2_hist->Fill(chi2);
    cache.pval_hist->Fill(prob(chi2, ndf));
    cache.reduced_chi2_hist->Fill(chi2 / ndf);
#endif  // TRACCC_HAVE_ROOT
}

}  // namespace traccc
