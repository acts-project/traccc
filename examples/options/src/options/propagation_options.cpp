/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/propagation_options.hpp"

// Detray include(s).
#include "detray/definitions/units.hpp"

// System include(s).
#include <limits>
#include <stdexcept>

namespace traccc {

/// Convenience namespace shorthand
namespace po = boost::program_options;

/// Type alias for the search window argument
using search_window_argument_type = std::vector<unsigned int>;
/// Name of the search window option
static const char* search_window_option = "search-window";

propagation_options::propagation_options(po::options_description& desc) {

    desc.add_options()("constraint-step-size-mm",
                       po::value(&(propagation.stepping.step_constraint))
                           ->default_value(std::numeric_limits<float>::max()),
                       "The constrained step size [mm]");
    desc.add_options()("overstep-tolerance-um",
                       po::value(&(propagation.navigation.overstep_tolerance))
                           ->default_value(-100.f),
                       "The overstep tolerance [um]");
    desc.add_options()("mask-tolerance-um",
                       po::value(&(propagation.navigation.mask_tolerance))
                           ->default_value(15.f),
                       "The mask tolerance [um]");
    desc.add_options()(search_window_option,
                       po::value<search_window_argument_type>()->multitoken(),
                       "Size of the grid surface search window");
    desc.add_options()(
        "rk-tolerance",
        po::value(&(propagation.stepping.rk_error_tol))->default_value(1e-4),
        "The Runge-Kutta stepper tolerance");
}

void propagation_options::read(const po::variables_map& vm) {

    propagation.stepping.step_constraint *= detray::unit<float>::mm;
    propagation.navigation.overstep_tolerance *= detray::unit<float>::um;
    propagation.navigation.mask_tolerance *= detray::unit<float>::um;

    // Set the search window parameter by hand, since boost::program_options
    // does not support std::array options directly.
    if (vm.count(search_window_option)) {
        const auto window =
            vm[search_window_option].as<search_window_argument_type>();
        if (window.size() != 2u) {
            throw std::invalid_argument(
                "Incorrect surface grid search window. Please provide two "
                "integer distances.");
        }
        propagation.navigation.search_window = {window[0], window[1]};
    } else {
        propagation.navigation.search_window = {0u, 0u};
    }
}

std::ostream& operator<<(std::ostream& out, const propagation_options& opt) {

    out << ">>> Propagation options <<<\n"
        << "  Constraint step size  : "
        << opt.propagation.stepping.step_constraint / detray::unit<float>::mm
        << " [mm]\n"
        << "  Overstep tolerance    : "
        << opt.propagation.navigation.overstep_tolerance /
               detray::unit<float>::um
        << " [um]\n"
        << "  Mask tolerance        : "
        << opt.propagation.navigation.mask_tolerance / detray::unit<float>::um
        << " [um]\n"
        << "  Search window         : "
        << opt.propagation.navigation.search_window[0] << " x "
        << opt.propagation.navigation.search_window[1] << "\n"
        << "  Runge-Kutta tolerance : " << opt.propagation.stepping.rk_error_tol
        << "\n";
    return out;
}

}  // namespace traccc
