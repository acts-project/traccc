/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// options
#include "traccc/options/common_options.hpp"

traccc::common_options::common_options(po::options_description& desc) {

    desc.add_options()("input-csv", "Use csv input file");
    desc.add_options()("input-binary", "Use binary input file");
    desc.add_options()("input-directory", po::value<std::string>()->required(),
                       "specify the directory of input data");
    desc.add_options()("events", po::value<unsigned int>()->required(),
                       "number of events");
    desc.add_options()("skip", po::value<int>()->default_value(0),
                       "number of events to skip");
    desc.add_options()("target-cells-per-partition",
                       po::value<unsigned short>()->default_value(1024),
                       "Number of cells to merge in a partition. Equal to the "
                       "number of threads multiplied by CELLS_PER_THREAD "
                       "defined in clusterization.");
    desc.add_options()("check-performance",
                       po::value<bool>()->default_value(false),
                       "generate performance result");
    desc.add_options()("perform-ambiguity-resolution",
                       po::value<bool>()->default_value(true),
                       "perform ambiguity resolution");
}

void traccc::common_options::read(const po::variables_map& vm) {

    if (vm.count("input-csv")) {
        input_data_format = traccc::data_format::csv;
    } else if (vm.count("input-binary")) {
        input_data_format = traccc::data_format::binary;
    }
    input_directory = vm["input-directory"].as<std::string>();
    events = vm["events"].as<unsigned int>();
    skip = vm["skip"].as<int>();
    target_cells_per_partition =
        vm["target-cells-per-partition"].as<unsigned short>();
    check_performance = vm["check-performance"].as<bool>();
    perform_ambiguity_resolution = vm["perform-ambiguity-resolution"].as<bool>();
}