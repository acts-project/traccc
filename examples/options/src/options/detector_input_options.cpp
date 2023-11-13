/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// options
#include "traccc/options/detector_input_options.hpp"

traccc::detector_input_options::detector_input_options(
    po::options_description& desc) {
    desc.add_options()("detector_file", po::value<std::string>()->required(),
                       "specify detector file");
    desc.add_options()("material_file", po::value<std::string>(),
                       "specify material file");
    desc.add_options()("grid_file", po::value<std::string>(),
                       "specify surface grid file");
}

void traccc::detector_input_options::read(const po::variables_map& vm) {

    detector_file = vm["detector_file"].as<std::string>();
    if (vm.count("material_file")) {
        material_file = vm["material_file"].as<std::string>();
    }
    if (vm.count("grid_file")) {
        grid_file = vm["grid_file"].as<std::string>();
    }
}