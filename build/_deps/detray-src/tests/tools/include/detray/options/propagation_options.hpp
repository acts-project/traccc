/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/propagator/propagation_config.hpp"

// Detray test include(s)
#include "detray/options/options_handling.hpp"

// Boost
#include "detray/options/boost_program_options.hpp"

// System include(s)
#include <stdexcept>

namespace detray::options {

/// Add options for the navigation
template <>
void add_options<detray::navigation::config>(
    boost::program_options::options_description &desc,
    const detray::navigation::config &cfg) {

    desc.add_options()(
        "search_window",
        boost::program_options::value<std::vector<dindex>>()->multitoken(),
        "Search window size for the grid")(
        "min_mask_tolerance",
        boost::program_options::value<float>()->default_value(
            cfg.min_mask_tolerance / unit<float>::mm),
        "Minimum mask tolerance [mm]")(
        "max_mask_tolerance",
        boost::program_options::value<float>()->default_value(
            cfg.max_mask_tolerance / unit<float>::mm),
        "Maximum mask tolerance [mm]")(
        "mask_tolerance_scalor",
        boost::program_options::value<float>()->default_value(
            cfg.mask_tolerance_scalor),
        "Mask tolerance scaling")(
        "overstep_tolerance",
        boost::program_options::value<float>()->default_value(
            cfg.overstep_tolerance / unit<float>::um),
        "Overstepping tolerance [um] NOTE: Must be negative!")(
        "path_tolerance",
        boost::program_options::value<float>()->default_value(
            cfg.path_tolerance / unit<float>::um),
        "Tol. to decide when a track is on surface [um]");
}

/// Add options for the track parameter transport
template <>
void add_options<detray::stepping::config>(
    boost::program_options::options_description &desc,
    const detray::stepping::config &cfg) {

    desc.add_options()("minimum_stepsize",
                       boost::program_options::value<float>()->default_value(
                           cfg.min_stepsize / unit<float>::mm),
                       "Minimum step size [mm]")(
        "step_contraint",
        boost::program_options::value<float>()->default_value(
            cfg.step_constraint / unit<float>::mm),
        "Maximum step size [mm]")(
        "rk-tolerance",
        boost::program_options::value<float>()->default_value(cfg.rk_error_tol /
                                                              unit<float>::mm),
        "The Runge-Kutta intergration error tolerance [mm]")(
        "path_limit",
        boost::program_options::value<float>()->default_value(cfg.path_limit /
                                                              unit<float>::m),
        "Maximum path length for a track [m]")("mean_energy_loss",
                                               "Use Bethe energy loss")(
        "covariance_transport", "Run the covariance transport")(
        "eloss_gradient", "Use energy loss gradient in Jacobian transport")(
        "bfield_gradient", "Use B-field gradient in Jacobian transport");
}

/// Add options for the detray propagation
template <>
void add_options<detray::propagation::config>(
    boost::program_options::options_description &desc,
    const detray::propagation::config &cfg) {

    add_options(desc, cfg.navigation);
    add_options(desc, cfg.stepping);
}

/// Configure the navigator
template <>
void configure_options<detray::navigation::config>(
    boost::program_options::variables_map &vm,
    detray::navigation::config &cfg) {

    // Local navigation search window
    if (vm.count("search_window")) {
        const auto window = vm["search_window"].as<std::vector<dindex>>();
        if (window.size() == 2u) {
            cfg.search_window = {window[0], window[1]};
        } else {
            throw std::invalid_argument(
                "Incorrect surface grid search window. Please provide two "
                "integer distances.");
        }
    }
    // Overstepping tolerance
    if (!vm["min_mask_tolerance"].defaulted()) {
        const float mask_tol{vm["min_mask_tolerance"].as<float>()};
        assert(mask_tol >= 0.f);

        cfg.min_mask_tolerance = mask_tol * unit<float>::mm;
    }
    if (!vm["max_mask_tolerance"].defaulted()) {
        const float mask_tol{vm["max_mask_tolerance"].as<float>()};
        assert(mask_tol >= 0.f);

        cfg.max_mask_tolerance = mask_tol * unit<float>::mm;
    }
    if (!vm["mask_tolerance_scalor"].defaulted()) {
        const float mask_tol_scalor{vm["mask_tolerance_scalor"].as<float>()};
        assert(mask_tol_scalor >= 0.f);

        cfg.mask_tolerance_scalor = mask_tol_scalor;
    }
    if (!vm["overstep_tolerance"].defaulted()) {
        const float overstep_tol{vm["overstep_tolerance"].as<float>()};
        assert(overstep_tol <= 0.f);

        cfg.overstep_tolerance = overstep_tol * unit<float>::um;
    }
    if (!vm["path_tolerance"].defaulted()) {
        const float path_tol{vm["path_tolerance"].as<float>()};
        assert(path_tol >= 0.f);

        cfg.path_tolerance = path_tol * unit<float>::um;
    }
}

/// Configure the stepper
template <>
void configure_options<detray::stepping::config>(
    boost::program_options::variables_map &vm, detray::stepping::config &cfg) {

    // Overstepping tolerance
    if (!vm["minimum_stepsize"].defaulted()) {
        const float min_step{vm["minimum_stepsize"].as<float>()};
        assert(min_step >= 0.f);

        cfg.min_stepsize = min_step * unit<float>::mm;
    }
    if (!vm["step_contraint"].defaulted()) {
        const float constraint{vm["step_contraint"].as<float>()};
        assert(constraint >= 0.f);

        cfg.step_constraint = constraint * unit<float>::mm;
    }
    if (!vm["rk-tolerance"].defaulted()) {
        const float err_tol{vm["rk-tolerance"].as<float>()};
        assert(err_tol >= 0.f);

        cfg.rk_error_tol = err_tol * unit<float>::mm;
    }
    if (!vm["path_limit"].defaulted()) {
        const float path_limit{vm["path_limit"].as<float>()};
        assert(path_limit > 0.f);

        cfg.path_limit = path_limit * unit<float>::m;
    }
    if (vm.count("covariance_transport")) {
        cfg.do_covariance_transport = true;
    }
    if (vm.count("mean_eloss")) {
        cfg.use_mean_loss = true;
    }
    if (vm.count("eloss_gradient")) {
        cfg.use_eloss_gradient = true;
    }
    if (vm.count("bfield_gradient")) {
        cfg.use_field_gradient = true;
    }
}

/// Configure the propagation
template <>
void configure_options<detray::propagation::config>(
    boost::program_options::variables_map &vm,
    detray::propagation::config &cfg) {

    configure_options(vm, cfg.navigation);
    configure_options(vm, cfg.stepping);
}

}  // namespace detray::options
