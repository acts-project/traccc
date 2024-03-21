/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/output_data.hpp"

// System include(s).
#include <stdexcept>

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

/// Description of this option group
static const char* description = "Output Data Options";

/// Type alias for the data format enumeration
using data_format_type = std::string;
/// Name of the data format option
static const char* data_format_option = "output-data-format";

output_data::output_data(boost::program_options::options_description& desc)
    : m_desc{description} {

    m_desc.add_options()(data_format_option,
                         po::value<data_format_type>()->default_value("csv"),
                         "Format of the output file(s)");
    m_desc.add_options()("output-directory",
                         po::value(&directory)->default_value(directory),
                         "Directory to store the output files");
    desc.add(m_desc);
}

void output_data::read(const boost::program_options::variables_map& vm) {

    // Decode the input data format.
    if (vm.count(data_format_option)) {
        const std::string input_format_string =
            vm[data_format_option].as<data_format_type>();
        if (input_format_string == "csv") {
            format = data_format::csv;
        } else if (input_format_string == "binary") {
            format = data_format::binary;
        } else if (input_format_string == "json") {
            format = data_format::json;
        } else {
            throw std::invalid_argument("Unknown input data format");
        }
    }
}

std::ostream& operator<<(std::ostream& out, const output_data& opt) {

    out << ">>> " << description << " <<<\n"
        << "  Output data format : " << opt.format << "\n"
        << "  Output directory   : " << opt.directory;
    return out;
}

}  // namespace traccc::opts
