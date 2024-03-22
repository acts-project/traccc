/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// options
#include "traccc/options/seeding_input_options.hpp"

// System include(s).
#include <iostream>

namespace traccc {

/// Convenience namespace shorthand
namespace po = boost::program_options;

seeding_input_options::seeding_input_options(po::options_description&) {}

void seeding_input_options::read(const po::variables_map&) {}

std::ostream& operator<<(std::ostream& out, const seeding_input_options&) {

    out << ">>> Seeding input options <<<";
    return out;
}

}  // namespace traccc
