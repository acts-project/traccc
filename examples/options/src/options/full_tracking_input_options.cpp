/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/full_tracking_input_options.hpp"

// System include(s).
#include <iostream>

namespace traccc {

/// Convenience namespace shorthand
namespace po = boost::program_options;

full_tracking_input_options::full_tracking_input_options(
    po::options_description& desc) {

    desc.add_options()("digitization-config-file",
                       po::value(&digitization_config_file)
                           ->default_value(digitization_config_file),
                       "specify the digitization configuration file");
}

void full_tracking_input_options::read(const po::variables_map&) {}

std::ostream& operator<<(std::ostream& out,
                         const full_tracking_input_options& opt) {

    out << ">>> Full tracking chain options <<<\n"
        << "  Digitization configuration file: "
        << opt.digitization_config_file;
    return out;
}

}  // namespace traccc
