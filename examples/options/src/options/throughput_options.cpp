/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/options/throughput_options.hpp"

// System include(s).
#include <iostream>
#include <stdexcept>

namespace traccc {

namespace po = boost::program_options;

throughput_options::throughput_options(po::options_description& desc) {

    desc.add_options()("input_data_format",
                       po::value<std::string>()->default_value("csv"),
                       "Format of the input file(s)");
    desc.add_options()("input_directory", po::value<std::string>()->required(),
                       "Directory holding the input files");
    desc.add_options()("detector_file", po::value<std::string>()->required(),
                       "Detector geometry description file");
    desc.add_options()("digitization_config_file",
                       po::value<std::string>()->required(),
                       "Digitization configuration file");
    desc.add_options()("loaded_events",
                       po::value<std::size_t>()->default_value(10),
                       "Number of input events to load");
    desc.add_options()("processed_events",
                       po::value<std::size_t>()->default_value(100),
                       "Number of events to process");
    desc.add_options()("cold_run_events",
                       po::value<std::size_t>()->default_value(10),
                       "Number of events to run 'cold'");
}

void throughput_options::read(const po::variables_map& vm) {

    // Set the input data format.
    if (vm.count("input_data_format")) {
        const std::string input_format_string =
            vm["input_data_format"].as<std::string>();
        if (input_format_string == "csv") {
            input_data_format = data_format::csv;
        } else if (input_format_string == "binary") {
            input_data_format = data_format::binary;
        } else {
            throw std::invalid_argument("Invalid input data format specified");
        }
    }

    // Set the rest of the options.
    input_directory = vm["input_directory"].as<std::string>();
    detector_file = vm["detector_file"].as<std::string>();
    digitization_config_file = vm["digitization_config_file"].as<std::string>();
    loaded_events = vm["loaded_events"].as<std::size_t>();
    processed_events = vm["processed_events"].as<std::size_t>();
    cold_run_events = vm["cold_run_events"].as<std::size_t>();
}

std::ostream& operator<<(std::ostream& out, const throughput_options& opt) {

    out << ">>> Throughput options <<<\n"
        << "Input data format   : " << opt.input_data_format << "\n"
        << "Input directory     : " << opt.input_directory << "\n"
        << "Detector geometry   : " << opt.detector_file << "\n"
        << "Digitization config : " << opt.digitization_config_file << "\n"
        << "Loaded event(s)     : " << opt.loaded_events << "\n"
        << "Cold run event(s)   : " << opt.cold_run_events << "\n"
        << "Processed event(s)  : " << opt.processed_events;
    return out;
}

}  // namespace traccc