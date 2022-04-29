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
    desc.add_options()("cell_directory", po::value<std::string>()->required(),
                       "specify the directory of cell files");
    desc.add_options()(
        "hit_directory", po::value<std::string>()->default_value(""),
        "specify the directory of hit files used for performance writer");
    desc.add_options()(
        "particle_directory", po::value<std::string>()->default_value(""),
        "specify the directory of particle files used for performance writer");
}

void traccc::full_tracking_input_config::read(const po::variables_map& vm) {
    detector_file = vm["detector_file"].as<std::string>();
    cell_directory = vm["cell_directory"].as<std::string>();
    hit_directory = vm["hit_directory"].as<std::string>();
    particle_directory = vm["particle_directory"].as<std::string>();
    check_seeding_performance =
        (!particle_directory.empty()) && (!hit_directory.empty());
}
