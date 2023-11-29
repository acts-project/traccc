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
    desc.add_options()("detector-file", po::value<std::string>()->required(),
                       "specify detector file");
    desc.add_options()("material-file",
                       po::value<std::string>()->default_value(""),
                       "specify material file");
    desc.add_options()("grid-file", po::value<std::string>()->default_value(""),
                       "specify surface grid file");
}

void traccc::detector_input_options::read(const po::variables_map& vm) {

    detector_file = vm["detector-file"].as<std::string>();
    if (vm.count("material-file")) {
        material_file = vm["material-file"].as<std::string>();
    }
    if (vm.count("grid-file")) {
        grid_file = vm["grid-file"].as<std::string>();
    }
}