/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/input_data.hpp"

// System include(s).
#include <iostream>
#include <stdexcept>

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

/// Type alias for the data format enumeration
using data_format_type = std::string;
/// Name of the data format option
static const char* data_format_option = "input-data-format";

input_data::input_data() : interface("Input Data Options") {

    m_desc.add_options()(
        "use-acts-geom-source",
        po::bool_switch(&use_acts_geom_source)->default_value(false),
        "Use acts geometry source");
    m_desc.add_options()(data_format_option,
                         po::value<data_format_type>()->default_value("csv"),
                         "Format of the input file(s)");
    m_desc.add_options()("input-directory",
                         po::value(&directory)->default_value(directory),
                         "Directory holding the input files");
    m_desc.add_options()("input-events",
                         po::value(&events)->default_value(events),
                         "Number of input events to process");
    m_desc.add_options()("input-skip", po::value(&skip)->default_value(skip),
                         "Number of input events to skip");
}

void input_data::read(const po::variables_map& vm) {

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

std::ostream& input_data::print_impl(std::ostream& out) const {

    out << "  Use ACTS geometry source      : "
        << (use_acts_geom_source ? "yes" : "no") << "\n"
        << "  Input data format             : " << format << "\n"
        << "  Input directory               : " << directory << "\n"
        << "  Number of input events        : " << events << "\n"
        << "  Number of input events to skip: " << skip;
    return out;
}

}  // namespace traccc::opts
