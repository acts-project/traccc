/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/options/options.hpp"

// Detray include(s).
#include "detray/definitions/units.hpp"
#include "detray/propagator/propagation_config.hpp"

// Boost
#include <boost/program_options.hpp>

// System
#include <limits>

namespace traccc {

namespace po = boost::program_options;

template <typename scalar_t>
struct propagation_options {

    detray::propagation::config<scalar_t> propagation{};

    propagation_options(po::options_description& desc) {
        desc.add_options()("constraint-step-size-mm",
                           po::value<scalar_t>()->default_value(
                               std::numeric_limits<scalar>::max()),
                           "The constrained step size [mm]");
        desc.add_options()("overstep-tolerance-um",
                           po::value<scalar_t>()->default_value(-100.f),
                           "The overstep tolerance [um]");
        desc.add_options()("mask-tolerance-um",
                           po::value<scalar_t>()->default_value(15.f),
                           "The mask tolerance [um]");
        desc.add_options()("search_window",
                           po::value<std::vector<unsigned int>>()->multitoken(),
                           "Size of the grid surface search window");
        desc.add_options()("rk-tolerance",
                           po::value<scalar_t>()->default_value(1e-4),
                           "The Runge-Kutta stepper tolerance");
    }

    void read(const po::variables_map& vm) {
        propagation.stepping.step_constraint =
            vm["constraint-step-size-mm"].as<scalar_t>() *
            detray::unit<scalar_t>::mm;
        propagation.navigation.overstep_tolerance =
            vm["overstep-tolerance-um"].as<scalar_t>() *
            detray::unit<scalar_t>::um;
        propagation.navigation.mask_tolerance =
            vm["mask-tolerance-um"].as<scalar_t>() * detray::unit<scalar_t>::um;
        propagation.stepping.rk_error_tol = vm["rk-tolerance"].as<scalar_t>();

        // Grid neighborhood size
        if (vm.count("search_window")) {
            const auto window =
                vm["search_window"].as<std::vector<unsigned int>>();
            if (window.size() != 2u) {
                throw std::invalid_argument(
                    "Incorrect surface grid search window. Please provide two "
                    "integer distances.");
            }
            propagation.navigation.search_window = {window[0], window[1]};
        } else {
            // default
            propagation.navigation.search_window = {0u, 0u};
        }
    }
};

}  // namespace traccc
