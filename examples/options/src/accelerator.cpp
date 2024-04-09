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

namespace traccc::opts {

accelerator::accelerator() : interface("Accelerator Options") {

    m_desc.add_options()("compare-with-cpu",
                         boost::program_options::bool_switch(&compare_with_cpu),
                         "Compare accelerator output with that of the CPU");
}

std::ostream& accelerator::print_impl(std::ostream& out) const {

    out << "  Compare with CPU results: " << (compare_with_cpu ? "yes" : "no");
    return out;
}

}  // namespace traccc::opts
