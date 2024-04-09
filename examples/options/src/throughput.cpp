/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/options/throughput.hpp"

// System include(s).
#include <iostream>

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

throughput::throughput() : interface("Throughput Measurement Options") {

    m_desc.add_options()(
        "processed-events",
        po::value(&processed_events)->default_value(processed_events),
        "Number of events to process");
    m_desc.add_options()(
        "cold-run-events",
        po::value(&cold_run_events)->default_value(cold_run_events),
        "Number of events to run 'cold'");
    m_desc.add_options()(
        "log-file", po::value(&log_file),
        "File where result logs will be printed (in append mode).");
}

std::ostream& throughput::print_impl(std::ostream& out) const {

    out << "  Cold run event(s) : " << cold_run_events << "\n"
        << "  Processed event(s): " << processed_events << "\n"
        << "  Log file          : " << log_file;
    return out;
}

}  // namespace traccc::opts
