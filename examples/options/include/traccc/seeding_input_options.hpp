/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Boost
#include <boost/program_options.hpp>

namespace traccc {

struct seeding_input_config {
    std::string detector_file;
    std::string hit_directory;
    std::string particle_directory;
    bool check_seeding_performance;
    unsigned int events;
    int skip;
};

namespace po = boost::program_options;

void add_seeding_input_options(po::options_description& desc) {

    desc.add_options()("detector_file", po::value<std::string>()->required(),
                       "specify detector file");
    desc.add_options()("hit_directory", po::value<std::string>()->required(),
                       "specify the directory of hit files");
    desc.add_options()("particle_directory",
                       po::value<std::string>()->default_value(""),
                       "specify the directory of particle files");
    desc.add_options()("events", po::value<unsigned int>()->required(),
                       "number of events");
    desc.add_options()("skip", po::value<int>()->default_value(0),
                       "number of events to skip");
}

seeding_input_config read_seeding_input_options(const po::variables_map& vm) {

    auto detector_file = vm["detector_file"].as<std::string>();
    auto hit_directory = vm["hit_directory"].as<std::string>();
    auto particle_directory = vm["particle_directory"].as<std::string>();
    auto events = vm["events"].as<unsigned int>();
    auto skip = vm["skip"].as<int>();
    auto check_seeding_performance = (particle_directory != "");

    return {detector_file,
            hit_directory,
            particle_directory,
            check_seeding_performance,
            events,
            skip};
}

}  // namespace traccc