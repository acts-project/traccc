/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/common_options.hpp"

// System include(s).
#include <iostream>

namespace traccc {

/// Convenience namespace shorthand
namespace po = boost::program_options;

/// Type alias for the data format enumeration
using data_format_type = std::string;
/// Name of the data format option
static const char* data_format_option = "input-data-format";

traccc::common_options::common_options(po::options_description& desc) {

    desc.add_options()(data_format_option,
                       po::value<data_format_type>()->default_value("csv"),
                       "Format of the input file(s)");
    desc.add_options()(
        "input-directory",
        po::value(&input_directory)->default_value(input_directory),
        "specify the directory of input data");
    desc.add_options()("events", po::value(&events)->default_value(events),
                       "number of events");
    desc.add_options()("skip", po::value(&skip)->default_value(skip),
                       "number of events to skip");
    desc.add_options()("target-cells-per-partition",
                       po::value(&target_cells_per_partition)
                           ->default_value(target_cells_per_partition),
                       "Number of cells to merge in a partition. Equal to the "
                       "number of threads multiplied by CELLS_PER_THREAD "
                       "defined in clusterization.");
    desc.add_options()("check-performance", po::bool_switch(&check_performance),
                       "generate performance result");
    desc.add_options()("perform-ambiguity-resolution",
                       po::value(&perform_ambiguity_resolution)
                           ->default_value(perform_ambiguity_resolution),
                       "perform ambiguity resolution");
}

void traccc::common_options::read(const po::variables_map& vm) {

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

std::ostream& operator<<(std::ostream& out, const common_options& opt) {

    out << ">>> Common options <<<\n"
        << "  Input data format            : " << opt.input_data_format << "\n"
        << "  Input directory              : " << opt.input_directory << "\n"
        << "  Events                       : " << opt.events << "\n"
        << "  Skipped events               : " << opt.skip << "\n"
        << "  Target cells per partition   : " << opt.target_cells_per_partition
        << "\n"
        << "  Check performance            : " << opt.check_performance << "\n"
        << "  Perform ambiguity resolution : "
        << opt.perform_ambiguity_resolution;
    return out;
}

}  // namespace traccc
