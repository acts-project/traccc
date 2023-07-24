/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// options
#include "traccc/options/full_tracking_input_options.hpp"

traccc::full_tracking_input_config::full_tracking_input_config(
    po::options_description& desc) {

    desc.add_options()("detector_file", po::value<std::string>()->required(),
                       "specify detector file");
    desc.add_options()("digitization_config_file",
                       po::value<std::string>()->required(),
                       "specify the digitization configuration file");
}

void traccc::full_tracking_input_config::read(const po::variables_map& vm) {
    detector_file = vm["detector_file"].as<std::string>();
    digitization_config_file = vm["digitization_config_file"].as<std::string>();
}
