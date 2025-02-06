/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/track_propagation.hpp"

#include "details/configuration_category.hpp"
#include "details/configuration_value.hpp"

// Detray include(s).
#include <detray/definitions/units.hpp>

// System include(s).
#include <format>
#include <limits>

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

track_propagation::track_propagation()
    : interface("Track Propagation Options") {
    m_search_window[0] = m_config.navigation.search_window[0];
    m_search_window[1] = m_config.navigation.search_window[1];

    m_desc.add_options()("constraint-step-size-mm",
                         po::value(&(m_config.stepping.step_constraint))
                             ->default_value(m_config.stepping.step_constraint),
                         "The constrained step size [mm]");
    m_desc.add_options()(
        "overstep-tolerance-um",
        po::value(&(m_config.navigation.overstep_tolerance))
            ->default_value(m_config.navigation.overstep_tolerance),
        "The overstep tolerance [um]");
    m_desc.add_options()(
        "min-mask-tolerance-mm",
        po::value(&(m_config.navigation.min_mask_tolerance))
            ->default_value(m_config.navigation.min_mask_tolerance),
        "The minimum mask tolerance [mm]");
    m_desc.add_options()(
        "max-mask-tolerance-mm",
        po::value(&(m_config.navigation.max_mask_tolerance))
            ->default_value(m_config.navigation.max_mask_tolerance),
        "The maximum mask tolerance [mm]");
    m_desc.add_options()(
        "search-window",
        po::value(&m_search_window)->default_value(m_search_window),
        "Size of the grid surface search window");
    m_desc.add_options()("rk-tolerance",
                         po::value(&(m_config.stepping.rk_error_tol))
                             ->default_value(m_config.stepping.rk_error_tol),
                         "The Runge-Kutta stepper tolerance");
}

void track_propagation::read(const po::variables_map &) {

    m_config.stepping.step_constraint *= detray::unit<float>::mm;
    m_config.navigation.overstep_tolerance *= detray::unit<float>::um;
    m_config.navigation.min_mask_tolerance *= detray::unit<float>::mm;
    m_config.navigation.max_mask_tolerance *= detray::unit<float>::mm;
    m_config.navigation.search_window = m_search_window;
}

track_propagation::operator detray::propagation::config() const {
    return m_config;
}

std::unique_ptr<configuration_printable> track_propagation::as_printable()
    const {

    auto cat_nav = std::make_unique<configuration_category>("Navigation");

    cat_nav->add_child(std::make_unique<configuration_value>(
        "Min mask tolerance",
        std::to_string(m_config.navigation.min_mask_tolerance /
                       detray::unit<float>::mm) +
            " mm"));
    cat_nav->add_child(std::make_unique<configuration_value>(
        "Max mask tolerance",
        std::to_string(m_config.navigation.max_mask_tolerance /
                       detray::unit<float>::mm) +
            " mm"));
    cat_nav->add_child(std::make_unique<configuration_value>(
        "Mask tolerance scalar",
        std::to_string(m_config.navigation.mask_tolerance_scalor)));
    cat_nav->add_child(std::make_unique<configuration_value>(
        "Path tolerance", std::to_string(m_config.navigation.path_tolerance /
                                         detray::unit<float>::um) +
                              " um"));
    cat_nav->add_child(std::make_unique<configuration_value>(
        "Overstep tolerance",
        std::to_string(m_config.navigation.overstep_tolerance /
                       detray::unit<float>::um) +
            " um"));
    cat_nav->add_child(std::make_unique<configuration_value>(
        "Search window",
        std::to_string(m_config.navigation.search_window[0]) + " x " +
            std::to_string(m_config.navigation.search_window[1])));

    auto cat_tsp = std::make_unique<configuration_category>("Transport");

    cat_tsp->add_child(std::make_unique<configuration_value>(
        "Min step size", std::to_string(m_config.stepping.min_stepsize /
                                        detray::unit<float>::mm) +
                             " mm"));
    cat_tsp->add_child(std::make_unique<configuration_value>(
        "Runge-Kutta tolerance", std::to_string(m_config.stepping.rk_error_tol /
                                                detray::unit<float>::mm) +
                                     " mm"));
    cat_tsp->add_child(std::make_unique<configuration_value>(
        "Max step updates", std::to_string(m_config.stepping.max_rk_updates)));
    cat_tsp->add_child(std::make_unique<configuration_value>(
        "Step size constraint",
        std::to_string(m_config.stepping.step_constraint /
                       detray::unit<float>::mm) +
            " mm"));
    cat_tsp->add_child(std::make_unique<configuration_value>(
        "Path limit",
        std::to_string(m_config.stepping.path_limit / detray::unit<float>::m) +
            " m"));
    cat_tsp->add_child(std::make_unique<configuration_value>(
        "Min step size", std::to_string(m_config.stepping.min_stepsize /
                                        detray::unit<float>::mm) +
                             " mm"));
    cat_tsp->add_child(std::make_unique<configuration_value>(
        "Enable Bethe energy loss",
        std::format("{}", m_config.stepping.use_mean_loss)));
    cat_tsp->add_child(std::make_unique<configuration_value>(
        "Enable covariance transport",
        std::format("{}", m_config.stepping.do_covariance_transport)));

    if (m_config.stepping.do_covariance_transport) {
        auto cat_cov =
            std::make_unique<configuration_category>("Covariance transport");

        cat_cov->add_child(std::make_unique<configuration_value>(
            "Enable energy loss gradient",
            std::format("{}", m_config.stepping.use_eloss_gradient)));
        cat_cov->add_child(std::make_unique<configuration_value>(
            "Enable B-field gradient",
            std::format("{}", m_config.stepping.use_field_gradient)));

        cat_tsp->add_child(std::move(cat_cov));
    }

    auto cat_geo = std::make_unique<configuration_category>("Geometry context");

    auto result = std::make_unique<configuration_category>(m_description);

    result->add_child(std::move(cat_nav));
    result->add_child(std::move(cat_tsp));
    result->add_child(std::move(cat_geo));

    return result;
}

}  // namespace traccc::opts
