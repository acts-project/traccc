/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/track_parameters.hpp"
#include "traccc/utils/helpers.hpp"

namespace traccc {

class res_plot_tool {

    public:
    /// @brief The nested configuration struct
    struct config {
        /// parameter sets to do plots
        std::vector<std::string> param_names = {"d0",    "z0",  "phi",
                                                "theta", "qop", "t"};

        /// Binning setups
        std::map<std::string, plot_helpers::binning> var_binning = {
            {"pull", plot_helpers::binning("pull", 100, -5, 5)},
            {"residual_d0",
             plot_helpers::binning("r_{d0} [mm]", 100, -0.5, 0.5)},
            {"residual_z0",
             plot_helpers::binning("r_{z0} [mm]", 100, -0.5, 0.5)},
            {"residual_phi",
             plot_helpers::binning("r_{#phi} [rad]", 100, -0.01, 0.01)},
            {"residual_theta",
             plot_helpers::binning("r_{#theta} [rad]", 100, -0.01, 0.01)},
            {"residual_qop",
             plot_helpers::binning("r_{q/p} [c/GeV]", 100, -0.1, 0.1)},
            {"residual_t",
             plot_helpers::binning("r_{t} [s]", 100, -1000, 1000)}};
    };

    /// @brief Nested Cache struct
    struct res_plot_cache {
        // Residuals and pulls for parameters
        std::map<std::string, TH1F*> residuals;
        std::map<std::string, TH1F*> pulls;
    };

    /// Constructor
    ///
    /// @param cfg Configuration struct
    res_plot_tool(const config& cfg);

    /// @brief book the resolution plots
    ///
    /// @param cache the cache for resolution plots
    void book(res_plot_cache& cache) const;

    /// @brief fill the cache
    ///
    /// @param cache the cache for resolution plots
    /// @param truth_param truth track parameter
    /// @param fit_param fitted track parameter
    void fill(res_plot_cache& cache, const bound_track_parameters& truth_param,
              const bound_track_parameters& fit_param) const;

    /// @brief write the resolution plots into ROOT
    ///
    /// @param cache the cache for resolution plots
    void write(const res_plot_cache& cache) const;

    /// @brief clear the cache
    ///
    /// @param cache the cache for resolution plots
    void clear(const res_plot_cache& cache) const;

    private:
    config m_cfg;  ///< The Config class
};

}  // namespace traccc