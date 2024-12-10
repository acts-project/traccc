/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Detray test include(s)
#include "detray/options/options_handling.hpp"
#include "detray/test/utils/detectors/build_toy_detector.hpp"

// Boost
#include "detray/options/boost_program_options.hpp"

// System include(s)
#include <stdexcept>
#include <string>

namespace detray::options {

/// Add options for the detray toy detector
template <>
void add_options<toy_det_config>(
    boost::program_options::options_description &desc,
    const toy_det_config &cfg) {

    desc.add_options()(
        "barrel_layers",
        boost::program_options::value<unsigned int>()->default_value(
            cfg.n_brl_layers()),
        "number of barrel layers [0-4]")(
        "endcap_layers",
        boost::program_options::value<unsigned int>()->default_value(
            cfg.n_edc_layers()),
        "number of endcap layers on either side [0-7]")(
        "homogeneous_material",
        "Generate homogeneous material description (default)")(
        "material_maps", "Generate material maps");
}

/// Configure the detray toy detector
template <>
void configure_options<toy_det_config>(
    boost::program_options::variables_map &vm, toy_det_config &cfg) {

    cfg.n_brl_layers(vm["barrel_layers"].as<unsigned int>());
    cfg.n_edc_layers(vm["endcap_layers"].as<unsigned int>());

    if (vm.count("homogeneous_material") && vm.count("material_maps")) {
        throw std::invalid_argument(
            "Please specify only one material description");
    }
    if (vm.count("homogeneous_material")) {
        cfg.use_material_maps(false);
    }
    if (vm.count("material_maps")) {
        cfg.use_material_maps(true);
    }
}

}  // namespace detray::options
