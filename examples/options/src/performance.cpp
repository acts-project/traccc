/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/performance.hpp"

// System include(s).
#include <iostream>

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

/// Description of this option group
static const char* description = "Performance Measurement Options";

performance::performance(po::options_description& desc) : m_desc{description} {

    m_desc.add_options()("check-performance", po::bool_switch(&run),
                         "Run performance checks");
    desc.add(m_desc);
}

void performance::read(const po::variables_map&) {}

std::ostream& operator<<(std::ostream& out, const performance& opts) {

    out << ">>> " << description << " <<<\n"
        << "  - Run performance checks : " << opts.run;
    return out;
}

}  // namespace traccc::opts
