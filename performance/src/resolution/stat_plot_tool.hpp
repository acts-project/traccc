/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "../utils/helpers.hpp"
#include "traccc/resolution/stat_plot_tool_config.hpp"

// Project include(s).
#include "traccc/edm/track_state.hpp"

namespace traccc {

class stat_plot_tool {

    public:
    /// @brief Nested Cache struct
    struct stat_plot_cache {
#ifdef TRACCC_HAVE_ROOT
        // Histogram for the number of DoFs
        std::unique_ptr<TH1> ndf_hist;
        // Histogram for the chi sqaure
        std::unique_ptr<TH1> chi2_hist;
        // Histogram for the pvalue
        std::unique_ptr<TH1> pval_hist;
#endif  // TRACCC_HAVE_ROOT
    };

    /// Constructor
    ///
    /// @param cfg Configuration struct
    stat_plot_tool(const stat_plot_tool_config& cfg);

    /// @brief book the statistics plots
    ///
    /// @param cache the cache for statistics plots
    void book(stat_plot_cache& cache) const;

    /// @brief fill the cache
    ///
    /// @param cache the cache for statistics plots
    /// @param fit_info fitting information that contains statistics
    void fill(stat_plot_cache& cache,
              const fitter_info<transform3>& fit_info) const;

    /// @brief write the statistics plots into ROOT
    ///
    /// @param cache the cache for statistics plots
    void write(const stat_plot_cache& cache) const;

    private:
    stat_plot_tool_config m_cfg;  ///< The Config class
};

}  // namespace traccc
