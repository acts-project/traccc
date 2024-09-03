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

performance::performance() : interface("Performance Measurement Options") {

    m_desc.add_options()("check-performance",
                         boost::program_options::bool_switch(&run),
                         "Run performance checks");

    m_desc.add_options()(
        "print-performance",
        boost::program_options::bool_switch(&print_performance),
        "Print small performance metrics on the terminal "
        "(valid/duplicate/fake tracks statistics)");
}

std::ostream& performance::print_impl(std::ostream& out) const {

    out << "  Run performance checks: " << (run ? "yes" : "no");
    out << "  Print performance metrics: "
        << (print_performance ? "yes" : "no");
    return out;
}

}  // namespace traccc::opts
