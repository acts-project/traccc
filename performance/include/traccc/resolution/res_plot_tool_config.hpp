/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/utils/helpers.hpp"

// System include(s).
#include <map>
#include <string>
#include <vector>

namespace traccc {

/// @brief Configuration structure for @c traccc::res_plot_tool
struct res_plot_tool_config {

    /// parameter sets to do plots
    std::vector<std::string> param_names = {"d0",    "z0",  "phi",
                                            "theta", "qop", "t"};

    /// Binning setups
    std::map<std::string, plot_helpers::binning> var_binning = {
        {"pull", plot_helpers::binning("pull", 100, -5, 5)},
        {"residual_d0", plot_helpers::binning("r_{d0} [mm]", 100, -0.5, 0.5)},
        {"residual_z0", plot_helpers::binning("r_{z0} [mm]", 100, -0.5, 0.5)},
        {"residual_phi",
         plot_helpers::binning("r_{#phi} [rad]", 100, -0.01, 0.01)},
        {"residual_theta",
         plot_helpers::binning("r_{#theta} [rad]", 100, -0.01, 0.01)},
        {"residual_qop",
         plot_helpers::binning("r_{q/p} [c/GeV]", 100, -0.1, 0.1)},
        {"residual_t", plot_helpers::binning("r_{t} [s]", 100, -1000, 1000)}};
};

}  // namespace traccc
