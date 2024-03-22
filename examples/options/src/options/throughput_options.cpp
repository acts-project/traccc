/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/options/throughput_options.hpp"

// System include(s).
#include <iostream>
#include <stdexcept>

namespace traccc {

/// Convenience namespace shorthand
namespace po = boost::program_options;

/// Type alias for the data format enumeration
using data_format_type = std::string;
/// Name of the data format option
static const char* data_format_option = "input-data-format";

throughput_options::throughput_options(po::options_description& desc) {

    desc.add_options()(data_format_option,
                       po::value<data_format_type>()->default_value("csv"),
                       "Format of the input file(s)");
    desc.add_options()(
        "input-directory",
        po::value(&input_directory)->default_value(input_directory),
        "Directory holding the input files");
    desc.add_options()("detector-file",
                       po::value(&detector_file)->default_value(detector_file),
                       "Detector geometry description file");
    desc.add_options()("digitization-config-file",
                       po::value(&digitization_config_file)
                           ->default_value(digitization_config_file),
                       "Digitization configuration file");
    desc.add_options()(
        "target-cells-per-partition",
        po::value(&target_cells_per_partition)
            ->default_value(target_cells_per_partition),
        "Average number of cells in a partition. Equal to the number of "
        "threads in the clusterization kernels multiplied by CELLS_PER_THREAD "
        "defined in clusterization.");
    desc.add_options()("loaded-events",
                       po::value(&loaded_events)->default_value(loaded_events),
                       "Number of input events to load");
    desc.add_options()(
        "processed-events",
        po::value(&processed_events)->default_value(processed_events),
        "Number of events to process");
    desc.add_options()(
        "cold-run-events",
        po::value(&cold_run_events)->default_value(cold_run_events),
        "Number of events to run 'cold'");
    desc.add_options()(
        "log-file", po::value(&log_file),
        "File where result logs will be printed (in append mode).");
}

void throughput_options::read(const po::variables_map& vm) {

    // Set the input data format.
    if (vm.count(data_format_option)) {
        const auto input_format_string =
            vm[data_format_option].as<data_format_type>();
        if (input_format_string == "csv") {
            input_data_format = data_format::csv;
        } else if (input_format_string == "binary") {
            input_data_format = data_format::binary;
        } else {
            throw std::invalid_argument("Invalid input data format specified");
        }
    }
}

std::ostream& operator<<(std::ostream& out, const throughput_options& opt) {

    out << ">>> Throughput options <<<\n"
        << "  Input data format          : " << opt.input_data_format << "\n"
        << "  Input directory            : " << opt.input_directory << "\n"
        << "  Detector geometry          : " << opt.detector_file << "\n"
        << "  Digitization config        : " << opt.digitization_config_file
        << "\n"
        << "  Target cells per partition : " << opt.target_cells_per_partition
        << "\n"
        << "  Loaded event(s)            : " << opt.loaded_events << "\n"
        << "  Cold run event(s)          : " << opt.cold_run_events << "\n"
        << "  Processed event(s)         : " << opt.processed_events << "\n"
        << "  Log_file                   : " << opt.log_file;
    return out;
}

}  // namespace traccc