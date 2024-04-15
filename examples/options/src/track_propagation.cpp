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
}

void track_propagation::read(const po::variables_map&) {

    config.stepping.step_constraint *= detray::unit<float>::mm;
    config.navigation.overstep_tolerance *= detray::unit<float>::um;
    config.navigation.mask_tolerance *= detray::unit<float>::um;
    config.navigation.search_window = m_search_window;
}

void track_propagation::setup(detray::propagation::config<float>& cfg) const {

    cfg = config;
    return;
}

void track_propagation::setup(detray::propagation::config<double>& cfg) const {

    cfg.stepping.min_stepsize = config.stepping.min_stepsize;
    cfg.stepping.rk_error_tol = config.stepping.rk_error_tol;
    cfg.stepping.step_constraint = config.stepping.step_constraint;
    cfg.stepping.path_limit = config.stepping.path_limit;
    cfg.stepping.max_rk_updates = config.stepping.max_rk_updates;
    cfg.stepping.use_mean_loss = config.stepping.use_mean_loss;
    cfg.stepping.use_eloss_gradient = config.stepping.use_eloss_gradient;
    cfg.stepping.use_field_gradient = config.stepping.use_field_gradient;
    cfg.stepping.do_covariance_transport =
        config.stepping.do_covariance_transport;

    cfg.navigation.mask_tolerance = config.navigation.mask_tolerance;
    cfg.navigation.on_surface_tolerance =
        config.navigation.on_surface_tolerance;
    cfg.navigation.overstep_tolerance = config.navigation.overstep_tolerance;
    cfg.navigation.search_window[0] = config.navigation.search_window[0];
    cfg.navigation.search_window[1] = config.navigation.search_window[1];
    return;
}

std::ostream& track_propagation::print_impl(std::ostream& out) const {

    out << "  Constraint step size : "
        << config.stepping.step_constraint / detray::unit<float>::mm
        << " [mm]\n"
        << "  Overstep tolerance   : "
        << config.navigation.overstep_tolerance / detray::unit<float>::um
        << " [um]\n"
        << "  Mask tolerance       : "
        << config.navigation.mask_tolerance / detray::unit<float>::um
        << " [um]\n"
        << "  Search window        : " << config.navigation.search_window[0]
        << " x " << config.navigation.search_window[1] << "\n"
        << "  Runge-Kutta tolerance: " << config.stepping.rk_error_tol;
    return out;
}

}  // namespace traccc::opts
