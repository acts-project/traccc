/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/options/options.hpp"

// Detray include(s).
#include "detray/definitions/units.hpp"

// Boost
#include <boost/program_options.hpp>

// System
#include <limits>

namespace traccc {

namespace po = boost::program_options;

template <typename scalar_t>
struct propagation_options {
    scalar_t step_constraint{std::numeric_limits<scalar_t>::max()};
    scalar_t overstep_tolerance{-10.f * detray::unit<scalar_t>::um};

    propagation_options(po::options_description& desc) {
        desc.add_options()("constraint-step-size-mm",
                           po::value<scalar_t>()->default_value(
                               std::numeric_limits<scalar>::max()),
                           "The constrained step size [mm]");
        desc.add_options()("overstep-tolerance-um",
                           po::value<scalar_t>()->default_value(-10.f),
                           "The overstep tolerance [um]");
    }

    void read(const po::variables_map& vm) {
        step_constraint = vm["constraint-step-size-mm"].as<scalar_t>() *
                          detray::unit<scalar_t>::mm;
        overstep_tolerance = vm["overstep-tolerance-um"].as<scalar_t>() *
                             detray::unit<scalar_t>::um;
    }
};

}  // namespace traccc