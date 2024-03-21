/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/accelerator.hpp"

// System include(s).
#include <iostream>
#include <stdexcept>

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

/// Description of this option group
static const char* description = "Accelerator Options";

accelerator::accelerator(po::options_description& desc) : m_desc{description} {

    m_desc.add_options()("compare-with-cpu", po::bool_switch(&compare_with_cpu),
                         "Compare accelerator output with that of the CPU");
    desc.add(m_desc);
}

void accelerator::read(const po::variables_map&) {}

std::ostream& operator<<(std::ostream& out, const accelerator& opt) {

    out << ">>> " << description << " <<<\n"
        << "  Compare with CPU results: "
        << (opt.compare_with_cpu ? "yes" : "no");
    return out;
}

}  // namespace traccc::opts
