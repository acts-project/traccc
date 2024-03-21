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

/// Description of this option group
static const char* description = "Track Propagation Options";

track_propagation::track_propagation(po::options_description& desc)
    : m_desc{description} {

    m_desc.add_options()("constraint-step-size-mm",
                         po::value(&(config.stepping.step_constraint))
                             ->default_value(std::numeric_limits<float>::max()),
                         "The constrained step size [mm]");
    m_desc.add_options()("overstep-tolerance-um",
                         po::value(&(config.navigation.overstep_tolerance))
                             ->default_value(-100.f),
                         "The overstep tolerance [um]");
    m_desc.add_options()(
        "mask-tolerance-um",
        po::value(&(config.navigation.mask_tolerance))->default_value(15.f),
        "The mask tolerance [um]");
    m_desc.add_options()(
        "search-window",
        po::value(&m_search_window)->default_value(m_search_window),
        "Size of the grid surface search window");
    m_desc.add_options()(
        "rk-tolerance",
        po::value(&(config.stepping.rk_error_tol))->default_value(1e-4),
        "The Runge-Kutta stepper tolerance");
    desc.add(m_desc);
}

void track_propagation::read(const po::variables_map&) {

    config.stepping.step_constraint *= detray::unit<float>::mm;
    config.navigation.overstep_tolerance *= detray::unit<float>::um;
    config.navigation.mask_tolerance *= detray::unit<float>::um;
    config.navigation.search_window = m_search_window;
}

std::ostream& operator<<(std::ostream& out, const track_propagation& opt) {

    out << ">>> " << description << " <<<\n"
        << "  Constraint step size  : "
        << opt.config.stepping.step_constraint / detray::unit<float>::mm
        << " [mm]\n"
        << "  Overstep tolerance    : "
        << opt.config.navigation.overstep_tolerance / detray::unit<float>::um
        << " [um]\n"
        << "  Mask tolerance        : "
        << opt.config.navigation.mask_tolerance / detray::unit<float>::um
        << " [um]\n"
        << "  Search window         : "
        << opt.config.navigation.search_window[0] << " x "
        << opt.config.navigation.search_window[1] << "\n"
        << "  Runge-Kutta tolerance : " << opt.config.stepping.rk_error_tol;
    return out;
}

}  // namespace traccc::opts
