/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/threading.hpp"

// System include(s).
#include <iostream>
#include <stdexcept>

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

/// Description of this option group
static const char* description = "Multi-Threading Options";

threading::threading(po::options_description& desc) : m_desc{description} {

    m_desc.add_options()("cpu-threads",
                         po::value(&threads)->default_value(threads),
                         "The number of CPU threads to use");
    desc.add(m_desc);
}

void threading::read(const po::variables_map&) {

    if (threads == 0) {
        throw std::invalid_argument{"Must use threads>0"};
    }
}

std::ostream& operator<<(std::ostream& out, const threading& opt) {

    out << ">>> " << description << " <<<\n"
        << "  CPU threads: " << opt.threads;
    return out;
}

}  // namespace traccc::opts
