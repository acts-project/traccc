/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// options
#include "traccc/options/seeding_input_options.hpp"

traccc::seeding_input_config::seeding_input_config(
    po::options_description& desc) {

    desc.add_options()("detector_file", po::value<std::string>()->required(),
                       "specify detector file");
    desc.add_options()("run_detray_geometry",
                       po::value<bool>()->default_value(false),
                       "generate performance result");
    desc.add_options()("check_performance",
                       po::value<bool>()->default_value(false),
                       "generate performance result");
}

void traccc::seeding_input_config::read(const po::variables_map& vm) {
    detector_file = vm["detector_file"].as<std::string>();
    run_detray_geometry = vm["run_detray_geometry"].as<bool>();
    check_performance = vm["check_performance"].as<bool>();
}
