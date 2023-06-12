/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/utils/helpers.hpp"

// System include(s).
#include <map>
#include <vector>

namespace traccc {

/// @brief Configuration structure for @c traccc::stat_plot_tool
struct stat_plot_tool_config {

    /// Binning setups
    std::map<std::string, plot_helpers::binning> var_binning = {
        {"ndf", plot_helpers::binning("ndf", 10, 0., 0.)},
        {"chi2", plot_helpers::binning("chi2", 100, 0., 50.)},
        {"reduced_chi2", plot_helpers::binning("chi2/ndf", 100, 0., 10.)},
        {"pval", plot_helpers::binning("pval", 100, 0., 1.)}};
};

}  // namespace traccc
