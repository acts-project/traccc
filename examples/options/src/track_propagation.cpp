/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/track_propagation.hpp"

// Detray include(s).
#include <detray/definitions/units.hpp>

// System include(s).
#include <limits>

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

track_propagation::track_propagation()
    : interface("Track Propagation Options") {

    m_desc.add_options()("constraint-step-size-mm",
                         po::value(&(config.stepping.step_constraint))
                             ->default_value(std::numeric_limits<float>::max()),
                         "The constrained step size [mm]");
    m_desc.add_options()("overstep-tolerance-um",
                         po::value(&(config.navigation.overstep_tolerance))
                             ->default_value(-100.f),
                         "The overstep tolerance [um]");
    m_desc.add_options()("min-mask-tolerance-mm",
                         po::value(&(config.navigation.min_mask_tolerance))
                             ->default_value(1e-5f),
                         "The minimum mask tolerance [mm]");
    m_desc.add_options()(
        "max-mask-tolerance-mm",
        po::value(&(config.navigation.max_mask_tolerance))->default_value(1.f),
        "The maximum mask tolerance [mm]");
    m_desc.add_options()(
        "search-window",
        po::value(&m_search_window)->default_value(m_search_window),
        "Size of the grid surface search window");
    m_desc.add_options()(
        "rk-tolerance",
        po::value(&(config.stepping.rk_error_tol))->default_value(1e-4f),
        "The Runge-Kutta stepper tolerance");
}

void track_propagation::read(const po::variables_map&) {

    config.stepping.step_constraint *= detray::unit<float>::mm;
    config.navigation.overstep_tolerance *= detray::unit<float>::um;
    config.navigation.min_mask_tolerance *= detray::unit<float>::mm;
    config.navigation.max_mask_tolerance *= detray::unit<float>::mm;
    config.navigation.search_window = m_search_window;
}

track_propagation::operator detray::propagation::config() const {
    return config;
}

std::ostream& track_propagation::print_impl(std::ostream& out) const {

    out << config;

    return out;
}

}  // namespace traccc::opts
