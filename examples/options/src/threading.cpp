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

threading::threading() : interface("Multi-Threading Options") {

    m_desc.add_options()(
        "cpu-threads",
        boost::program_options::value(&threads)->default_value(threads),
        "The number of CPU threads to use");
}

void threading::read(const boost::program_options::variables_map&) {

    if (threads == 0) {
        throw std::invalid_argument{"Must use threads>0"};
    }
}

std::ostream& threading::print_impl(std::ostream& out) const {

    out << "  CPU threads: " << threads;
    return out;
}

}  // namespace traccc::opts
