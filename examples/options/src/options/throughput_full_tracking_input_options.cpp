/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// options
#include "traccc/options/throughput_full_tracking_input_options.hpp"

namespace traccc {

throughput_full_tracking_input_config::throughput_full_tracking_input_config(
    po::options_description& desc) {

    desc.add_options()("input-csv", "Use csv input file");
    desc.add_options()("input-binary", "Use binary input file");
    desc.add_options()("input_directory", po::value<std::string>()->required(),
                       "specify the directory of input data");
    desc.add_options()("detector_file", po::value<std::string>()->required(),
                       "specify detector file");
    desc.add_options()("digitization_config_file",
                       po::value<std::string>()->required(),
                       "specify the digitization configuration file");
    desc.add_options()("loaded_events", po::value<int>()->default_value(10),
                       "specify number of events to load");
    desc.add_options()("processed_events", po::value<int>()->default_value(100),
                       "specify number of events to process");
}

void throughput_full_tracking_input_config::read(const po::variables_map& vm) {
    if (vm.count("input-csv")) {
        input_data_format = data_format::csv;
    } else if (vm.count("input-binary")) {
        input_data_format = data_format::binary;
    }

    input_directory = vm["input_directory"].as<std::string>();
    detector_file = vm["detector_file"].as<std::string>();
    digitization_config_file = vm["digitization_config_file"].as<std::string>();
    loaded_events = vm["loaded_events"].as<int>();
    processed_events = vm["processed_events"].as<int>();
}

std::ostream& operator<<(std::ostream& out,
                         const throughput_full_tracking_input_config& cfg) {
    out << cfg.detector_file << " " << cfg.input_directory << " "
        << cfg.loaded_events << " " << cfg.processed_events;
    return out;
}

}  // namespace traccc